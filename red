#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SPGD locking script with Red Pitaya hardware and simulation backends.

This script implements a simultaneous perturbation stochastic gradient descent
(SPGD) controller that can either drive a simulated interferometer or a real
Red Pitaya board.  The hardware mode reads the detector signal from the fast
ADC ``IN1`` input and generates the control signal through the ``OUT1`` DAC.

The goal is to keep the combining efficiency in the 95–98 % range while
providing real-time visualisation that mirrors the simulation view.  The code is
organised so that both the simulator and the hardware backend expose the same
minimal interface which allows the plotting code to remain unchanged.
"""

from __future__ import annotations

import argparse
import math
import random
import socket
import statistics
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover - optional dependency for plotting
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - fallback for headless tests
    MATPLOTLIB_AVAILABLE = False
    plt = None  # type: ignore
    GridSpec = object  # type: ignore
    FuncAnimation = object  # type: ignore

try:  # pragma: no cover - optional dependency for spectral estimates
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - very small fallback for tests
    class _SimpleNumpy:
        pi = math.pi

        class random:
            @staticmethod
            def normal(loc: float = 0.0, scale: float = 1.0) -> float:
                return random.gauss(loc, scale)

            @staticmethod
            def uniform(low: float, high: float) -> float:
                return random.uniform(low, high)

        @staticmethod
        def array(data: Iterable[float]):
            return list(data)

        @staticmethod
        def clip(value: float, minimum: float, maximum: float) -> float:
            return max(minimum, min(maximum, value))

        @staticmethod
        def cos(x: float) -> float:
            return math.cos(x)

        @staticmethod
        def exp(z: complex) -> complex:
            return math.e ** z

        @staticmethod
        def linspace(start: float, stop: float, num: int, endpoint: bool = True):
            if num <= 1:
                return [start]
            step = (stop - start) / (num - 1 if endpoint else num)
            return [start + i * step for i in range(num)]

        @staticmethod
        def mean(values: Iterable[float]) -> float:
            data = list(values)
            return sum(data) / len(data) if data else 0.0

        @staticmethod
        def std(values: Iterable[float]) -> float:
            data = list(values)
            if len(data) < 2:
                return 0.0
            mu = _SimpleNumpy.mean(data)
            return math.sqrt(sum((x - mu) ** 2 for x in data) / len(data))

        @staticmethod
        def abs(value: complex) -> float:
            return abs(value)

        @staticmethod
        def angle(z: complex) -> float:
            return math.atan2(z.imag, z.real)

        @staticmethod
        def max(values: Iterable[float]) -> float:
            return max(values)

        @staticmethod
        def min(values: Iterable[float]) -> float:
            return min(values)

    np = _SimpleNumpy()  # type: ignore

try:  # pragma: no cover - optional dependency
    from scipy import signal
except ModuleNotFoundError:  # pragma: no cover - degrade gracefully
    signal = None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def wrap_to_pi(value: float) -> float:
    """Wrap an angle to the [-pi, pi] interval."""
    return math.atan2(math.sin(value), math.cos(value))


def mean_square(values: Iterable[float]) -> float:
    """Return the mean square of the iterable."""
    data = list(values)
    if not data:
        return 0.0
    return sum(v * v for v in data) / len(data)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


@dataclass
class SPGDParameters:
    gain: float = 1.8
    learning_rate: float = 0.18
    perturbation: float = 0.08
    momentum: float = 0.85
    bias_gain: float = 0.08
    bias_decay: float = 0.86
    min_learning_rate: float = 0.02
    max_learning_rate: float = 0.55
    min_perturbation: float = 0.01
    max_perturbation: float = 0.18
    efficiency_target: float = 97.0
    efficiency_floor: float = 95.0
    emergency_threshold: float = 92.0
    max_phase_step: float = 0.22
    velocity_clip: float = 0.28


class SPGDController:
    """Adaptive SPGD controller."""

    def __init__(self, params: Optional[SPGDParameters] = None):
        self.params = params or SPGDParameters()
        self.estimated_phase = random.uniform(-math.pi, math.pi)
        self.velocity = 0.0
        self.gradient_ema = 0.0
        self.bias_integrator = 0.0
        self.locked = False
        self.emergency_mode = False
        self.lock_count = 0
        self.unlock_count = 0
        self._eff_history: Deque[float] = deque(maxlen=50)

    # ------------------------------------------------------------------
    def _update_learning_rate(self, efficiency: float, gradient: float) -> None:
        params = self.params
        grad_mag = abs(gradient)
        if efficiency > params.efficiency_target + 0.5 and grad_mag < 0.01:
            params.learning_rate = max(params.min_learning_rate, params.learning_rate * 0.92)
        elif efficiency < params.efficiency_floor or grad_mag > 0.3:
            params.learning_rate = min(params.max_learning_rate, params.learning_rate * 1.15)
        elif efficiency < params.efficiency_target:
            params.learning_rate = min(params.max_learning_rate, params.learning_rate * 1.05)
        else:
            params.learning_rate = max(params.min_learning_rate, params.learning_rate * 0.98)

    # ------------------------------------------------------------------
    def update(self, gradient: float, efficiency: float) -> float:
        """Update the internal phase estimate."""
        p = self.params
        self._eff_history.append(efficiency)

        if efficiency < p.emergency_threshold and self.locked:
            self.emergency_mode = True
            self.locked = False
            self.unlock_count += 1
            p.learning_rate = p.max_learning_rate
            p.perturbation = p.max_perturbation
            self.velocity = 0.0
        elif efficiency > p.efficiency_target and self.emergency_mode:
            self.emergency_mode = False

        # Adaptive perturbation magnitude
        if efficiency > p.efficiency_target + 0.5:
            p.perturbation = max(p.min_perturbation, p.perturbation * 0.92)
        elif efficiency > p.efficiency_target:
            p.perturbation = max(p.min_perturbation, p.perturbation * 0.96)
        elif efficiency < p.efficiency_floor:
            p.perturbation = min(p.max_perturbation, p.perturbation * 1.18)
        else:
            p.perturbation = min(p.max_perturbation, p.perturbation * 1.05)

        # Bias integrator keeps slow drifts centred
        error = p.efficiency_target - efficiency
        self.bias_integrator = (p.bias_decay * self.bias_integrator +
                                (1 - p.bias_decay) * error)
        bias_term = self.bias_integrator * p.bias_gain

        self.gradient_ema = 0.85 * self.gradient_ema + 0.15 * gradient
        corrected_grad = self.gradient_ema + bias_term
        self._update_learning_rate(efficiency, corrected_grad)

        raw_step = p.learning_rate * corrected_grad
        self.velocity = (p.momentum * self.velocity + raw_step)
        self.velocity = float(np.clip(self.velocity, -p.velocity_clip, p.velocity_clip))
        phase_increment = float(np.clip(p.gain * self.velocity,
                                        -p.max_phase_step, p.max_phase_step))
        self.estimated_phase = wrap_to_pi(self.estimated_phase + phase_increment)

        if len(self._eff_history) == self._eff_history.maxlen:
            mean_eff = float(np.mean(self._eff_history))
            std_eff = float(np.std(self._eff_history))
            if mean_eff > p.efficiency_target and std_eff < 0.8:
                if not self.locked:
                    self.locked = True
                    self.lock_count += 1
            else:
                if self.locked and efficiency < p.efficiency_floor - 0.5:
                    self.locked = False
                    self.unlock_count += 1
        return self.estimated_phase


# ---------------------------------------------------------------------------
# Red Pitaya SCPI communication
# ---------------------------------------------------------------------------


class RedPitayaSCPIClient:
    """Minimal SCPI client with sane defaults for Red Pitaya."""

    def __init__(self, host: str = "192.168.10.2", port: int = 5000,
                 timeout: float = 5.0, eol: bytes = b"\r\n"):
        self.host = host
        self.port = port
        self.timeout = float(timeout)
        self.eol = eol
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._buffer = b""

    # ------------------------------------------------------------------
    def connect(self) -> None:
        if self._sock is None:
            sock = socket.create_connection((self.host, self.port), self.timeout)
            sock.settimeout(self.timeout)
            self._sock = sock

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
                self._buffer = b""

    # ------------------------------------------------------------------
    @contextmanager
    def session(self):  # pragma: no cover - convenience wrapper
        try:
            self.connect()
            yield self
        finally:
            self.close()

    # ------------------------------------------------------------------
    def write(self, command: str) -> None:
        self.connect()
        assert self._sock is not None
        data = command.strip().encode("ascii", errors="ignore") + self.eol
        with self._lock:
            self._sock.sendall(data)

    # ------------------------------------------------------------------
    def _readline(self) -> str:
        newline = b"\n"
        assert self._sock is not None
        deadline = time.time() + self.timeout
        with self._lock:
            while True:
                if newline in self._buffer:
                    line, self._buffer = self._buffer.split(newline, 1)
                    return line.decode("ascii", errors="ignore").rstrip("\r")
                if time.time() > deadline:
                    raise TimeoutError("SCPI timeout")
                chunk = self._sock.recv(4096)
                if not chunk:
                    raise ConnectionError("Connection lost")
                self._buffer += chunk

    # ------------------------------------------------------------------
    def query(self, command: str) -> str:
        self.write(command)
        return self._readline()

    # ------------------------------------------------------------------
    def identify(self) -> str:
        try:
            return self.query("*IDN?")
        except Exception as exc:  # pragma: no cover - hardware specific
            raise ConnectionError(f"Failed to query Red Pitaya identity: {exc}") from exc

    # ------------------------------------------------------------------
    def configure_dc_output(self, channel: int = 1) -> None:
        """Configure the selected output channel for DC control."""
        self.write("GEN:RST")
        self.write(f"SOUR{channel}:FUNC DC")
        self.write(f"SOUR{channel}:VOLT 0")
        self.write(f"SOUR{channel}:VOLT:OFFS 0")
        self.write(f"OUTPUT{channel}:STATE ON")

    # ------------------------------------------------------------------
    def set_output_offset(self, channel: int, voltage: float) -> None:
        self.write(f"SOUR{channel}:VOLT:OFFS {voltage}")

    # ------------------------------------------------------------------
    def acquire_samples(self, source: int = 1, count: int = 16384,
                         decimation: int = 1, averaging: bool = True,
                         timeout: float = 1.0) -> List[float]:
        self.write("ACQ:RST")
        self.write(f"ACQ:DEC {decimation}")
        self.write(f"ACQ:AVG {'ON' if averaging else 'OFF'}")
        self.write("ACQ:START")
        self.write("ACQ:TRIG NOW")

        deadline = time.time() + timeout
        while time.time() < deadline:
            state = self.query("ACQ:TRIG:STAT?").strip().upper()
            if state in {"TD", "TRIGGERED"}:
                break
            time.sleep(0.01)
        else:
            raise TimeoutError("Red Pitaya acquisition timeout")

        raw = self.query(f"ACQ:SOUR{source}:DATA? {count}")
        return self._parse_samples(raw)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_samples(raw: str) -> List[float]:
        data = raw.strip()
        if data.startswith("{") and data.endswith("}"):
            data = data[1:-1]
        values: List[float] = []
        for item in data.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                values.append(float(item))
            except ValueError:
                continue
        return values


# ---------------------------------------------------------------------------
# Simulation backend
# ---------------------------------------------------------------------------


class SPGDSimulator:
    """Simple two-beam coherent combining simulator."""

    def __init__(self, controller: Optional[SPGDController] = None,
                 dt: float = 0.0005):
        self.controller = controller or SPGDController()
        self.dt = dt
        self.time_history: Deque[float] = deque(maxlen=60000)
        self.intensity_history: Deque[float] = deque(maxlen=60000)
        self.efficiency_history: Deque[float] = deque(maxlen=60000)
        self.phase_error_history: Deque[float] = deque(maxlen=60000)
        self.true_phase_history: Deque[float] = deque(maxlen=60000)
        self.estimated_phase_history: Deque[float] = deque(maxlen=60000)
        self.intensity_buffer_for_noise: Deque[float] = deque(maxlen=2048)
        self.power_beam1 = 1.0
        self.power_beam2 = 1.0
        self.max_intensity = 4.0
        self.iteration = 0
        self.time = 0.0
        self.true_phase = random.uniform(-math.pi, math.pi)
        self.drift_rate = 0.03
        self.phase_noise = 0.01
        self.measurement_noise = 0.002

    # ------------------------------------------------------------------
    def _calculate_intensity(self, phase_error: float) -> float:
        p1 = self.power_beam1 * (1 + random.gauss(0.0, 0.0015))
        p2 = self.power_beam2 * (1 + random.gauss(0.0, 0.0015))
        intensity = p1 + p2 + 2 * math.sqrt(p1 * p2) * math.cos(phase_error)
        return max(0.0, intensity + random.gauss(0.0, self.measurement_noise))

    # ------------------------------------------------------------------
    def _calculate_efficiency(self, intensity: float) -> float:
        return 100.0 * intensity / self.max_intensity

    # ------------------------------------------------------------------
    def _measure_gradient(self, perturbation: float) -> float:
        phase = self.controller.estimated_phase
        intensity_plus = self._calculate_intensity(phase + perturbation)
        intensity_minus = self._calculate_intensity(phase - perturbation)
        return (intensity_plus - intensity_minus) / (2 * perturbation)

    # ------------------------------------------------------------------
    def step(self, substeps: int = 1) -> Dict[str, float]:
        data: Dict[str, float] = {}
        for _ in range(max(1, substeps)):
            self.iteration += 1
            self.time += self.dt
            self.true_phase = wrap_to_pi(
                self.true_phase + random.gauss(0.0, self.phase_noise) +
                random.gauss(0.0, self.drift_rate) * self.dt)

            gradient = self._measure_gradient(self.controller.params.perturbation)
            phase_error_before = wrap_to_pi(self.controller.estimated_phase - self.true_phase)
            intensity_before = self._calculate_intensity(phase_error_before)
            efficiency_before = self._calculate_efficiency(intensity_before)
            new_phase = self.controller.update(gradient, efficiency_before)
            phase_error_after = wrap_to_pi(new_phase - self.true_phase)
            current_intensity = self._calculate_intensity(phase_error_after)
            current_efficiency = self._calculate_efficiency(current_intensity)

            self.time_history.append(self.time)
            self.intensity_history.append(current_intensity)
            self.efficiency_history.append(current_efficiency)
            self.phase_error_history.append(phase_error_after)
            self.true_phase_history.append(self.true_phase)
            self.estimated_phase_history.append(new_phase)
            self.intensity_buffer_for_noise.append(current_intensity)

            data = {
                "time": self.time,
                "intensity": current_intensity,
                "efficiency": current_efficiency,
                "phase_error": phase_error_after,
                "estimated_phase": new_phase,
                "locked": self.controller.locked,
                "perturbation": self.controller.params.perturbation,
                "gain_scale": self.controller.params.gain,
            }
        return data

    # ------------------------------------------------------------------
    def calculate_noise_spectrum(self) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        if signal is None or len(self.intensity_buffer_for_noise) < 256:
            return None, None
        data = np.array(list(self.intensity_buffer_for_noise))
        data = data - np.mean(data)
        if len(self.time_history) < len(data):
            return None, None
        times = list(self.time_history)[-len(data):]
        span = times[-1] - times[0]
        if span <= 0:
            return None, None
        dt = span / (len(data) - 1)
        freq, psd = signal.welch(data, fs=1.0 / dt, nperseg=min(512, len(data)))
        mean_intensity = np.mean(data) + self.max_intensity / 2
        if mean_intensity <= 0:
            rin = psd
        else:
            rin = psd / (mean_intensity ** 2)
        return freq.tolist(), rin.tolist()


# ---------------------------------------------------------------------------
# Hardware backend
# ---------------------------------------------------------------------------


class RedPitayaHardwareLock:
    """SPGD locking backend for a Red Pitaya."""

    def __init__(self, client: Optional[RedPitayaSCPIClient] = None,
                 controller: Optional[SPGDController] = None,
                 control_channel: int = 1, measurement_channel: int = 1,
                 samples_per_measurement: int = 4096, decimation: int = 8,
                 settle_time: float = 0.002, output_limits: Tuple[float, float] = (-1.0, 1.0)):
        self.client = client or RedPitayaSCPIClient()
        self.controller = controller or SPGDController()
        self.control_channel = control_channel
        self.measurement_channel = measurement_channel
        self.samples_per_measurement = samples_per_measurement
        self.decimation = decimation
        self.settle_time = settle_time
        self.output_limits = output_limits
        self.current_phase = self.controller.estimated_phase
        self.start_time: Optional[float] = None
        self.iteration = 0
        self.dt = settle_time * 2
        self.time_history: Deque[float] = deque(maxlen=20000)
        self.intensity_history: Deque[float] = deque(maxlen=20000)
        self.efficiency_history: Deque[float] = deque(maxlen=20000)
        self.phase_error_history: Deque[float] = deque(maxlen=20000)
        self.true_phase_history: Deque[float] = deque(maxlen=20000)
        self.estimated_phase_history: Deque[float] = deque(maxlen=20000)
        self.intensity_buffer_for_noise: Deque[float] = deque(maxlen=4096)
        self.max_intensity = 1.0
        self.min_intensity = 1e-6
        self.reported_efficiency: Optional[float] = None

    # ------------------------------------------------------------------
    def initialise(self) -> None:
        print("Connecting to Red Pitaya...")
        self.client.connect()
        identity = self.client.identify()
        print(f"Device: {identity}")
        self.client.configure_dc_output(self.control_channel)
        self._calibrate()
        self.start_time = time.time()

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        try:  # pragma: no cover - hardware specific
            self.client.write(f"OUTPUT{self.control_channel}:STATE OFF")
        finally:
            self.client.close()

    # ------------------------------------------------------------------
    def _phase_to_voltage(self, phase: float) -> float:
        phase = wrap_to_pi(phase)
        low, high = self.output_limits
        span = high - low
        normalized = (phase + math.pi) / (2 * math.pi)
        voltage = low + span * normalized
        return float(np.clip(voltage, low, high))

    # ------------------------------------------------------------------
    def _apply_phase(self, phase: float) -> None:
        voltage = self._phase_to_voltage(phase)
        self.client.set_output_offset(self.control_channel, voltage)
        time.sleep(self.settle_time)

    # ------------------------------------------------------------------
    def _measure_intensity(self) -> float:
        samples = self.client.acquire_samples(
            source=self.measurement_channel,
            count=self.samples_per_measurement,
            decimation=self.decimation,
            averaging=True,
        )
        if not samples:
            return self.min_intensity
        return mean_square(samples)

    # ------------------------------------------------------------------
    def _efficiency_from_intensity(self, intensity: float) -> float:
        span = max(self.max_intensity - self.min_intensity, 1e-6)
        efficiency = (intensity - self.min_intensity) / span
        efficiency = max(0.0, min(1.0, efficiency))
        return efficiency * 100.0

    # ------------------------------------------------------------------
    def _calibrate(self, scan_points: int = 90) -> None:
        print("Scanning output range to estimate intensity bounds...")
        phases = np.linspace(-math.pi, math.pi, scan_points, endpoint=False)
        max_intensity = -1.0
        min_intensity = float("inf")
        best_phase = self.current_phase

        for phase in phases:
            self._apply_phase(phase)
            intensity = self._measure_intensity()
            if intensity > max_intensity:
                max_intensity = intensity
                best_phase = phase
            if intensity < min_intensity:
                min_intensity = intensity

        self.max_intensity = max(max_intensity, 1e-3)
        self.min_intensity = max(min_intensity, 1e-6)
        self.current_phase = best_phase
        self.controller.estimated_phase = best_phase
        self._apply_phase(best_phase)
        print(f"Calibration done. Imax={self.max_intensity:.4f}, Imin={self.min_intensity:.4f}")

    # ------------------------------------------------------------------
    def _measure_gradient(self) -> Tuple[float, float, float]:
        perturbation = self.controller.params.perturbation
        plus_phase = wrap_to_pi(self.current_phase + perturbation)
        minus_phase = wrap_to_pi(self.current_phase - perturbation)

        self._apply_phase(plus_phase)
        intensity_plus = self._measure_intensity()
        self._apply_phase(minus_phase)
        intensity_minus = self._measure_intensity()

        # Restore working point
        self._apply_phase(self.current_phase)
        baseline_intensity = self._measure_intensity()
        baseline_efficiency = self._efficiency_from_intensity(baseline_intensity)

        gradient = (intensity_plus - intensity_minus) / (2 * perturbation)
        return gradient, baseline_intensity, baseline_efficiency

    # ------------------------------------------------------------------
    def step(self, substeps: int = 1) -> Dict[str, float]:
        data: Dict[str, float] = {}
        for _ in range(max(1, substeps)):
            if self.start_time is None:
                self.start_time = time.time()
            self.iteration += 1
            gradient, baseline_intensity, baseline_efficiency = self._measure_gradient()
            new_phase = self.controller.update(gradient, baseline_efficiency)
            self.current_phase = new_phase
            self._apply_phase(new_phase)
            intensity = self._measure_intensity()
            efficiency = self._efficiency_from_intensity(intensity)

            # Smooth efficiency to reduce ADC noise influence
            if self.reported_efficiency is None:
                smoothed_eff = efficiency
            else:
                smoothed_eff = 0.7 * self.reported_efficiency + 0.3 * efficiency
            self.reported_efficiency = smoothed_eff

            timestamp = time.time() - self.start_time
            phase_error = wrap_to_pi(new_phase)  # treat reference as zero

            self.time_history.append(timestamp)
            self.intensity_history.append(intensity)
            self.efficiency_history.append(smoothed_eff)
            self.phase_error_history.append(phase_error)
            self.true_phase_history.append(0.0)
            self.estimated_phase_history.append(new_phase)
            self.intensity_buffer_for_noise.append(intensity)

            data = {
                "time": timestamp,
                "intensity": intensity,
                "efficiency": smoothed_eff,
                "phase_error": phase_error,
                "estimated_phase": new_phase,
                "locked": self.controller.locked,
                "perturbation": self.controller.params.perturbation,
                "gain_scale": self.controller.params.gain,
            }
        return data

    # ------------------------------------------------------------------
    def calculate_noise_spectrum(self) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        if signal is None or len(self.intensity_buffer_for_noise) < 256:
            return None, None
        data = np.array(list(self.intensity_buffer_for_noise))
        data = data - np.mean(data)
        if len(self.time_history) < len(data):
            return None, None
        times = list(self.time_history)[-len(data):]
        span = times[-1] - times[0]
        if span <= 0:
            return None, None
        dt = span / (len(data) - 1)
        freq, psd = signal.welch(data, fs=1.0 / dt, nperseg=min(512, len(data)))
        mean_intensity = np.mean(list(self.intensity_buffer_for_noise))
        if mean_intensity <= 0:
            rin = psd
        else:
            rin = psd / (mean_intensity ** 2)
        return freq.tolist(), rin.tolist()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


class RealtimeDisplay:
    """Realtime visualisation shared by both backends."""

    def __init__(self, system, frame_interval: float = 0.1):
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("Matplotlib is required for plotting")

        self.system = system
        dt = getattr(system, "dt", frame_interval)
        if dt <= 0:
            dt = frame_interval
        self.steps_per_frame = max(1, int(round(frame_interval / dt)))
        self.frame_interval = frame_interval

        self.fig = plt.figure(figsize=(14, 8))
        self.fig.suptitle('SPGD Coherent Combining', fontsize=14, fontweight='bold')
        gs = GridSpec(3, 2, figure=self.fig, hspace=0.35, wspace=0.3)

        self.ax_efficiency = self.fig.add_subplot(gs[0, 0])
        self.ax_phase = self.fig.add_subplot(gs[1, 0])
        self.ax_intensity = self.fig.add_subplot(gs[2, 0])
        self.ax_hist = self.fig.add_subplot(gs[0, 1])
        self.ax_noise = self.fig.add_subplot(gs[1, 1])
        self.ax_text = self.fig.add_subplot(gs[2, 1])

        self._setup_axes()

    # ------------------------------------------------------------------
    def _setup_axes(self) -> None:
        self.ax_efficiency.set_ylabel('Efficiency (%)')
        self.ax_efficiency.set_ylim(0, 105)
        target = self.system.controller.params.efficiency_target
        floor = self.system.controller.params.efficiency_floor
        self.ax_efficiency.axhline(target, color='green', linestyle='--', alpha=0.5)
        self.ax_efficiency.axhline(floor, color='red', linestyle='--', alpha=0.5)
        (self.line_efficiency,) = self.ax_efficiency.plot([], [], 'b-', lw=2)

        self.ax_phase.set_ylabel('Phase error (rad)')
        self.ax_phase.set_ylim(-math.pi, math.pi)
        (self.line_phase,) = self.ax_phase.plot([], [], 'r-', lw=1.5)

        self.ax_intensity.set_xlabel('Time (s)')
        self.ax_intensity.set_ylabel('Intensity (a.u.)')
        (self.line_intensity,) = self.ax_intensity.plot([], [], 'k-', lw=1.5)

        self.ax_hist.set_title('Efficiency distribution')
        self.ax_noise.set_title('Intensity noise spectrum')
        self.ax_noise.set_xlabel('Frequency (Hz)')
        self.ax_noise.set_ylabel('RIN (1/Hz)')
        self.ax_noise.set_xscale('log')
        self.ax_noise.set_yscale('log')
        (self.line_noise,) = self.ax_noise.plot([], [], 'g-')

        self.ax_text.axis('off')
        self.text_box = self.ax_text.text(0.02, 0.95, '', va='top', family='monospace')

    # ------------------------------------------------------------------
    def update(self, _frame):
        data = self.system.step(substeps=self.steps_per_frame)
        times = list(getattr(self.system, 'time_history', []))
        efficiencies = list(getattr(self.system, 'efficiency_history', []))
        phases = list(getattr(self.system, 'phase_error_history', []))
        intensities = list(getattr(self.system, 'intensity_history', []))

        if times:
            self.line_efficiency.set_data(times, efficiencies)
            self.ax_efficiency.set_xlim(max(0.0, times[-1] - 10.0), times[-1] + 0.5)
            self.line_phase.set_data(times, phases)
            self.ax_phase.set_xlim(self.ax_efficiency.get_xlim())
            self.line_intensity.set_data(times, intensities)
            self.ax_intensity.set_xlim(self.ax_efficiency.get_xlim())

        if len(efficiencies) > 10:
            self.ax_hist.clear()
            self.ax_hist.set_title('Efficiency distribution')
            self.ax_hist.hist(efficiencies[-200:], bins=20, color='steelblue', edgecolor='black')
            self.ax_hist.axvline(self.system.controller.params.efficiency_target, color='green', linestyle='--')
            self.ax_hist.axvline(self.system.controller.params.efficiency_floor, color='red', linestyle='--')

        freq, rin = self.system.calculate_noise_spectrum()
        if freq and rin and len(freq) > 1:
            self.line_noise.set_data(freq[1:], rin[1:])
            self.ax_noise.relim()
            self.ax_noise.autoscale_view()

        if efficiencies:
            stats = efficiencies[-200:]
            text = [f"Samples: {len(efficiencies)}"]
            text.append(f"Mean eff: {statistics.mean(stats):6.2f} %")
            if len(stats) > 1:
                text.append(f"Std dev : {statistics.pstdev(stats):6.2f} %")
            text.append(f"Current : {efficiencies[-1]:6.2f} %")
            text.append(f"Perturb : {self.system.controller.params.perturbation:6.4f}")
            text.append(f"Learning: {self.system.controller.params.learning_rate:6.4f}")
            text.append(f"Locked  : {self.system.controller.locked}")
            self.text_box.set_text("\n".join(text))
        return [self.line_efficiency, self.line_phase, self.line_intensity, self.line_noise]

    # ------------------------------------------------------------------
    def run(self) -> None:
        ani = FuncAnimation(self.fig, self.update,
                            interval=int(self.frame_interval * 1000), blit=False)
        plt.show()


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


def run_simulation(args) -> None:
    controller = SPGDController()
    simulator = SPGDSimulator(controller)
    if args.plot:
        display = RealtimeDisplay(simulator, frame_interval=args.frame_interval)
        display.run()
    else:
        for _ in range(args.iterations):
            data = simulator.step()
            if (_ + 1) % 500 == 0:
                print(f"t={data['time']:.3f}s eff={data['efficiency']:.2f}% phase={data['phase_error']:.3f}rad")


def run_hardware(args) -> None:
    controller = SPGDController()
    client = RedPitayaSCPIClient(host=args.host, port=args.port, timeout=args.timeout)
    system = RedPitayaHardwareLock(client=client, controller=controller,
                                   control_channel=args.control_channel,
                                   measurement_channel=args.measurement_channel,
                                   samples_per_measurement=args.samples,
                                   decimation=args.decimation,
                                   settle_time=args.settle_time,
                                   output_limits=(args.output_min, args.output_max))
    system.initialise()
    try:
        if args.plot:
            display = RealtimeDisplay(system, frame_interval=args.frame_interval)
            display.run()
        else:
            end_time = time.time() + args.duration if args.duration else None
            while end_time is None or time.time() < end_time:
                data = system.step()
                print(f"t={data['time']:.2f}s eff={data['efficiency']:.2f}% phase={data['phase_error']:.3f}rad")
    finally:
        system.shutdown()


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='mode', required=True)

    sim_parser = subparsers.add_parser('simulate', help='Run the internal simulator')
    sim_parser.add_argument('--iterations', type=int, default=6000, help='Number of iterations to run')
    sim_parser.add_argument('--plot', action='store_true', help='Show real-time plots')
    sim_parser.add_argument('--frame-interval', type=float, default=0.05, help='Animation frame interval in seconds')
    sim_parser.set_defaults(func=run_simulation)

    hw_parser = subparsers.add_parser('hardware', help='Run with a Red Pitaya connected')
    hw_parser.add_argument('--host', type=str, default='192.168.10.2', help='Red Pitaya hostname or IP address')
    hw_parser.add_argument('--port', type=int, default=5000, help='SCPI port')
    hw_parser.add_argument('--timeout', type=float, default=5.0, help='Socket timeout in seconds')
    hw_parser.add_argument('--control-channel', type=int, default=1, help='Analog output channel used for control (OUTx)')
    hw_parser.add_argument('--measurement-channel', type=int, default=1, help='Analog input channel used for intensity (INx)')
    hw_parser.add_argument('--samples', type=int, default=4096, help='Samples acquired for each measurement')
    hw_parser.add_argument('--decimation', type=int, default=8, help='ADC decimation factor')
    hw_parser.add_argument('--settle-time', type=float, default=0.002, help='Settling delay after changing the control output')
    hw_parser.add_argument('--output-min', type=float, default=-1.0, help='Minimum control voltage for OUTx')
    hw_parser.add_argument('--output-max', type=float, default=1.0, help='Maximum control voltage for OUTx')
    hw_parser.add_argument('--duration', type=float, default=None, help='Optional duration limit in seconds')
    hw_parser.add_argument('--plot', action='store_true', help='Show real-time plots')
    hw_parser.add_argument('--frame-interval', type=float, default=0.1, help='Animation frame interval in seconds')
    hw_parser.set_defaults(func=run_hardware)

    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    args.func(args)


if __name__ == '__main__':  # pragma: no cover - CLI entry point
    main()
