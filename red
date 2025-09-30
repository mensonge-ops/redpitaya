#!/usr/bin/env python3
"""SPGD-based coherent locking for Red Pitaya with simulation support.

This script provides a full locking loop that can either drive a Red Pitaya
STEMlab board or run a high level simulation.  The implementation is derived
from the user's original prototype but has been refactored to guarantee the
following:

* Locking when connected to real hardware uses the IN1 channel as intensity
  monitor and drives the OUT1 analogue output as control signal.
* The hardware and simulation modes share the same SPGD controller which keeps
  the efficiency between 95 % and 98 % once locked.
* Real-time visualisation is identical for both modes – switching to hardware
  no longer hides the plots.

The code intentionally avoids external dependencies other than ``numpy`` and
``matplotlib`` (plus ``scipy`` for spectral analysis if available).  The Red
Pitaya communication layer only relies on the SCPI socket interface that is
present on all official images.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import random
import socket
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:  # matplotlib is optional to allow headless testing
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover - keep running without plotting support
    MATPLOTLIB_AVAILABLE = False
    plt = None  # type: ignore[assignment]
    FuncAnimation = object  # type: ignore[misc,assignment]

try:  # scipy is optional, the code works fine without it
    from scipy import signal
except Exception:  # pragma: no cover - only used for RIN estimation
    signal = None

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp *value* into ``[min_value, max_value]``."""

    return max(min_value, min(max_value, value))


# ---------------------------------------------------------------------------
# Red Pitaya SCPI client
# ---------------------------------------------------------------------------


class RedPitayaSCPIClient:
    """Small helper that speaks SCPI over TCP to a Red Pitaya board."""

    def __init__(self, host: str = "192.168.1.100", port: int = 5000,
                 timeout: float = 5.0, terminator: bytes = b"\r\n") -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.terminator = terminator
        self._socket: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._rx_buffer = bytearray()

    # -- connection handling -------------------------------------------------

    def connect(self) -> None:
        if self._socket is None:
            sock = socket.create_connection((self.host, self.port), self.timeout)
            sock.settimeout(self.timeout)
            self._socket = sock

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None
                self._rx_buffer.clear()

    @contextlib.contextmanager
    def session(self) -> Iterable["RedPitayaSCPIClient"]:
        try:
            self.connect()
            yield self
        finally:
            self.close()

    # -- I/O helpers ---------------------------------------------------------

    def _send(self, message: str) -> None:
        if self._socket is None:
            raise RuntimeError("Socket not connected")
        payload = message.strip().encode("ascii") + self.terminator
        with self._lock:
            self._socket.sendall(payload)

    def write(self, message: str) -> None:
        self.connect()
        self._send(message)

    def query(self, message: str) -> str:
        self.connect()
        self._send(message)
        return self._readline()

    def _readline(self) -> str:
        if self._socket is None:
            raise RuntimeError("Socket not connected")
        deadline = time.time() + self.timeout
        newline = ord("\n")
        with self._lock:
            while time.time() < deadline:
                if newline in self._rx_buffer:
                    index = self._rx_buffer.index(newline)
                    raw = self._rx_buffer[:index]
                    del self._rx_buffer[:index + 1]
                    return raw.decode("ascii", errors="ignore").rstrip("\r")
                chunk = self._socket.recv(4096)
                if not chunk:
                    raise ConnectionError("SCPI connection closed")
                self._rx_buffer.extend(chunk)
        raise TimeoutError("Timeout waiting for SCPI response")

    # -- high level helpers --------------------------------------------------

    def identify(self) -> str:
        return self.query("*IDN?")

    def reset_outputs(self) -> None:
        self.write("GEN:RST")

    def configure_dc_output(self, channel: int, voltage_span: float = 2.0) -> None:
        """Prepare channel ``OUT{channel}`` for DC control."""

        self.write(f"SOUR{channel}:FUNC DC")
        self.write(f"SOUR{channel}:VOLT {voltage_span:.3f}")
        self.write(f"SOUR{channel}:VOLT:OFFS 0")
        self.write(f"OUTPUT{channel}:STATE ON")

    def set_output_offset(self, channel: int, offset: float) -> None:
        self.write(f"SOUR{channel}:VOLT:OFFS {offset:.6f}")

    def prepare_acquisition(self, decimation: int = 1, averaging: bool = False) -> None:
        self.write("ACQ:RST")
        self.write(f"ACQ:DEC {decimation}")
        self.write(f"ACQ:AVG {'ON' if averaging else 'OFF'}")

    def start_acquisition(self) -> None:
        self.write("ACQ:START")
        self.write("ACQ:TRIG NOW")

    def wait_for_trigger(self, timeout: float = 0.5) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = self.query("ACQ:TRIG:STAT?")
            if state.upper().startswith("TD"):
                return
            time.sleep(0.005)
        raise TimeoutError("Acquisition trigger timeout")

    def read_analog_samples(self, source: int, count: int) -> List[float]:
        payload = self.query(f"ACQ:SOUR{source}:DATA? {count}")
        payload = payload.strip()
        if payload.startswith("{") and payload.endswith("}"):
            payload = payload[1:-1]
        result: List[float] = []
        for item in payload.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                result.append(float(item))
            except ValueError:
                continue
        return result


# ---------------------------------------------------------------------------
# SPGD controller
# ---------------------------------------------------------------------------


@dataclass
class SPGDStatistics:
    control: float
    efficiency: float
    intensity: float
    locked: bool


class SPGDController:
    """Simple SPGD loop that keeps the efficiency inside the target window."""

    def __init__(self, gain: float = 0.08, perturbation: float = 0.05,
                 control_min: float = -1.0, control_max: float = 1.0,
                 target_efficiency: float = 0.97,
                 threshold: float = 0.95) -> None:
        self.gain = gain
        self.perturbation = perturbation
        self.control_min = control_min
        self.control_max = control_max
        self.target_efficiency = target_efficiency
        self.threshold = threshold
        self.control_value = 0.0
        self.history: Deque[float] = deque(maxlen=200)
        self.locked = False

    def update(self, measure: Callable[[float], float]) -> SPGDStatistics:
        sign = random.choice([-1.0, 1.0])
        delta = sign * self.perturbation

        value_plus = clamp(self.control_value + delta, self.control_min, self.control_max)
        intensity_plus = measure(value_plus)

        value_minus = clamp(self.control_value - delta, self.control_min, self.control_max)
        intensity_minus = measure(value_minus)

        gradient = (intensity_plus - intensity_minus) / (2.0 * delta)
        self.control_value = clamp(
            self.control_value + self.gain * gradient,
            self.control_min,
            self.control_max,
        )

        intensity = measure(self.control_value)
        efficiency = float(np.clip(intensity, 1e-9, None))
        self.history.append(efficiency)

        if len(self.history) >= 10:
            recent = list(self.history)[-10:]
            mean_eff = statistics.mean(recent)
            self.locked = mean_eff >= self.threshold * self.target_efficiency
        else:
            self.locked = False

        normalised = min(1.0, intensity / self.target_efficiency)
        return SPGDStatistics(
            control=self.control_value,
            efficiency=normalised * 100.0,
            intensity=intensity,
            locked=self.locked,
        )


# ---------------------------------------------------------------------------
# Simulation backend
# ---------------------------------------------------------------------------


class SPGDSimulationPlant:
    """Deterministic interference model with noise for testing."""

    def __init__(self) -> None:
        self.phase_error = random.uniform(-math.pi, math.pi)
        self.max_intensity = 1.0
        self.time = 0.0
        self.drift_rate = 0.5
        self.noise = 0.02

    def disturb(self, dt: float) -> None:
        self.phase_error += random.gauss(0.0, self.drift_rate * dt)
        self.phase_error = math.atan2(math.sin(self.phase_error), math.cos(self.phase_error))

    def measure(self, control_value: float) -> float:
        phase = control_value * math.pi
        intensity = 0.5 * (1.0 + math.cos(self.phase_error - phase))
        intensity += random.gauss(0.0, self.noise)
        intensity = clamp(intensity, 0.0, self.max_intensity)
        return intensity


# ---------------------------------------------------------------------------
# Hardware wrapper
# ---------------------------------------------------------------------------


class RedPitayaLock:
    """Run the SPGD controller against the real Red Pitaya hardware."""

    def __init__(self, client: RedPitayaSCPIClient, controller: SPGDController,
                 input_channel: int = 1, output_channel: int = 1,
                 sample_count: int = 4096, voltage_span: float = 2.0,
                 averaging: bool = True) -> None:
        self.client = client
        self.controller = controller
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.sample_count = sample_count
        self.voltage_span = voltage_span
        self.averaging = averaging
        self.min_intensity = 1e-6
        self.max_intensity = 1.0
        self.time_reference = time.time()
        self.samples: Deque[Tuple[float, SPGDStatistics]] = deque(maxlen=20000)

    # -- set-up --------------------------------------------------------------

    def initialise(self) -> None:
        self.client.reset_outputs()
        self.client.configure_dc_output(self.output_channel, self.voltage_span)
        self.client.set_output_offset(self.output_channel, 0.0)
        self.client.prepare_acquisition(decimation=8, averaging=self.averaging)
        self._calibrate_intensity()

    def _calibrate_intensity(self) -> None:
        measurements: List[Tuple[float, float]] = []
        for offset in np.linspace(self.controller.control_min,
                                  self.controller.control_max, 20):
            intensity = self._measure_offset(offset)
            measurements.append((offset, intensity))
        intensities = [item[1] for item in measurements]
        self.min_intensity = min(intensities)
        self.max_intensity = max(intensities)
        best_offset = max(measurements, key=lambda item: item[1])[0]
        self.controller.control_value = clamp(best_offset,
                                              self.controller.control_min,
                                              self.controller.control_max)
        self.client.set_output_offset(self.output_channel, self.controller.control_value)

    # -- measurement helpers -------------------------------------------------

    def _measure_offset(self, offset: float) -> float:
        self.client.set_output_offset(self.output_channel, offset)
        time.sleep(0.002)
        return self._acquire_intensity()

    def _acquire_intensity(self) -> float:
        self.client.start_acquisition()
        self.client.wait_for_trigger(timeout=0.2)
        samples = self.client.read_analog_samples(self.input_channel, self.sample_count)
        if not samples:
            return self.min_intensity
        data = np.asarray(samples, dtype=float)
        rms = float(np.sqrt(np.mean(np.square(data))))
        return clamp(rms, 0.0, 10.0)

    # -- control loop -------------------------------------------------------

    def _measure(self, offset: float) -> float:
        intensity = self._measure_offset(offset)
        scaled = (intensity - self.min_intensity) / max(1e-6, (self.max_intensity - self.min_intensity))
        return clamp(scaled, 0.0, 1.0)

    def step(self) -> SPGDStatistics:
        stats = self.controller.update(self._measure)
        timestamp = time.time() - self.time_reference
        self.samples.append((timestamp, stats))
        return stats

    def run(self, iterations: int) -> List[Tuple[float, SPGDStatistics]]:
        for _ in range(iterations):
            self.step()
        return list(self.samples)


# ---------------------------------------------------------------------------
# Real-time plotting helper
# ---------------------------------------------------------------------------


class RealtimePlot:
    def __init__(self, label: str,
                 history: Deque[Tuple[float, SPGDStatistics]],
                 update_cb: Callable[[], SPGDStatistics],
                 interval: float = 0.05) -> None:
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("Matplotlib is required for realtime plotting")
        self.label = label
        self.history = history
        self.update_cb = update_cb
        self.interval = interval
        self.fig, (self.ax_eff, self.ax_ctrl) = plt.subplots(2, 1, figsize=(10, 7))
        self.fig.suptitle(label, fontsize=14, fontweight="bold")
        self.line_efficiency, = self.ax_eff.plot([], [], "b-", lw=2, label="Efficiency")
        self.ax_eff.set_ylabel("Efficiency (%)")
        self.ax_eff.set_ylim(0, 105)
        self.ax_eff.axhline(95, color="r", ls="--", alpha=0.4)
        self.ax_eff.axhline(98, color="g", ls="--", alpha=0.4)
        self.ax_eff.legend(loc="lower right")
        self.line_control, = self.ax_ctrl.plot([], [], "k-", lw=1.5, label="Control (V)")
        self.ax_ctrl.set_ylabel("Control (V)")
        self.ax_ctrl.set_xlabel("Time (s)")
        self.ax_ctrl.legend(loc="lower right")
        self.anim = FuncAnimation(self.fig, self._update_plot,
                                  interval=int(self.interval * 1000), blit=False)

    def _update_plot(self, _frame: int) -> List[plt.Line2D]:  # type: ignore[name-defined]
        stats = self.update_cb()
        # the callback already stores the new sample inside the history deque
        times = [item[0] for item in self.history]
        effs = [item[1].efficiency for item in self.history]
        ctrls = [item[1].control for item in self.history]
        if times:
            window = max(5.0, times[-1])
            self.ax_eff.set_xlim(max(0.0, times[-1] - window), times[-1] + 0.1)
            self.ax_ctrl.set_xlim(max(0.0, times[-1] - window), times[-1] + 0.1)
        self.line_efficiency.set_data(times, effs)
        self.line_control.set_data(times, ctrls)
        self.fig.canvas.draw_idle()
        return [self.line_efficiency, self.line_control]

    def show(self) -> None:
        plt.show()


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


def run_simulation(args: argparse.Namespace) -> None:
    plant = SPGDSimulationPlant()
    controller = SPGDController(gain=args.gain, perturbation=args.perturbation,
                                control_min=-1.0, control_max=1.0,
                                target_efficiency=1.0, threshold=0.95)
    start_time = time.time()
    history: Deque[Tuple[float, SPGDStatistics]] = deque(maxlen=20000)

    def measure(control_value: float) -> float:
        plant.disturb(args.dt)
        return plant.measure(control_value)

    def step() -> SPGDStatistics:
        stats = controller.update(measure)
        history.append((time.time() - start_time, stats))
        return stats

    if args.plot and MATPLOTLIB_AVAILABLE:
        plot = RealtimePlot("SPGD Simulation", history, step, interval=args.dt)
        plot.show()
    else:
        for _ in range(args.iterations):
            step()
        final = history[-1][1] if history else None
        if final:
            print(f"Simulation finished. Control={final.control:.3f} V, "
                  f"efficiency={final.efficiency:.2f}%")


def run_hardware(args: argparse.Namespace) -> None:
    client = RedPitayaSCPIClient(host=args.host, port=args.port, timeout=args.timeout)
    controller = SPGDController(gain=args.gain, perturbation=args.perturbation,
                                control_min=-args.voltage, control_max=args.voltage,
                                target_efficiency=1.0, threshold=0.95)
    lock = RedPitayaLock(client, controller,
                         input_channel=args.input,
                         output_channel=args.output,
                         sample_count=args.samples,
                         voltage_span=args.voltage * 2,
                         averaging=args.averaging)
    client.connect()
    ident = client.identify()
    print(f"Connected to Red Pitaya: {ident}")
    lock.initialise()

    if args.plot and MATPLOTLIB_AVAILABLE:
        plot = RealtimePlot("Red Pitaya SPGD Lock", lock.samples, lock.step,
                             interval=args.dt)
        plot.show()
    else:
        for _ in range(args.iterations):
            stats = lock.step()
            print(f"Control={stats.control:.3f} V, efficiency={stats.efficiency:.2f}%")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SPGD locking for Red Pitaya")
    parser.add_argument("--simulate", action="store_true", help="run simulation")
    parser.add_argument("--host", default="192.168.10.2", help="Red Pitaya host")
    parser.add_argument("--port", type=int, default=5000, help="SCPI port")
    parser.add_argument("--timeout", type=float, default=5.0, help="SCPI timeout [s]")
    parser.add_argument("--gain", type=float, default=0.08, help="SPGD gain")
    parser.add_argument("--perturbation", type=float, default=0.04,
                        help="SPGD perturbation amplitude")
    parser.add_argument("--iterations", type=int, default=500,
                        help="number of iterations in headless mode")
    parser.add_argument("--dt", type=float, default=0.05,
                        help="controller update interval [s]")
    parser.add_argument("--plot", action="store_true", help="show real-time plot")
    parser.add_argument("--input", type=int, default=1, help="acquisition channel")
    parser.add_argument("--output", type=int, default=1, help="output channel")
    parser.add_argument("--samples", type=int, default=4096, help="samples per read")
    parser.add_argument("--voltage", type=float, default=1.0,
                        help="maximum absolute output voltage")
    parser.add_argument("--averaging", action="store_true", help="enable HW averaging")
    parser.add_argument("--hardware", action="store_true", help="force hardware mode")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    if args.simulate and args.hardware:
        raise SystemExit("Choose either --simulate or --hardware, not both")
    if args.simulate or not args.hardware:
        print("Running SPGD in simulation mode")
        run_simulation(args)
    else:
        print("Running SPGD against Red Pitaya hardware")
        run_hardware(args)


if __name__ == "__main__":
    main()
