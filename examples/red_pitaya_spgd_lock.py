#!/usr/bin/env python3
"""Two-channel coherent locking on a Red Pitaya STEMlab 125-14 with SPGD.

This script implements a two-channel simultaneous perturbation stochastic
gradient descent (SPGD) controller that can drive a pair of actuators via the
analog outputs of a Red Pitaya STEMlab 125-14 in order to maximise the detected
combined intensity of two ultrashort pulses.  One pulse is assumed to run free
while the second is steered by the controller.  The detected intensity and the
resulting coherent synthesis efficiency are displayed in real time and the
controller can optionally search for the best locking point automatically.

The script contains a lightweight simulation backend so that the full control
loop can be exercised on a development machine without any hardware attached.
After convergence the script estimates the closed-loop transfer function as well
as the residual intensity-noise power spectral density.

Example usage with hardware::

    python examples/red_pitaya_spgd_lock.py --host 192.168.1.100 \
        --iterations 600 --gain 0.08 --perturbation 0.04

Example usage in simulation mode::

    python examples/red_pitaya_spgd_lock.py --simulate --plot

Configuration parameters can be stored in a JSON file and loaded with
``--config path/to/file.json``.  The repository ships with the file
``examples/red_pitaya_spgd_lock_config.json`` as a starting point.  Command-line
arguments override options loaded from JSON.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import socket
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # Optional dependency used for live plotting.
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except Exception:  # pragma: no cover - plotting is optional
    plt = None
    FuncAnimation = None


class BackendError(RuntimeError):
    """Raised when the hardware backend reports an unexpected condition."""


class Backend:
    """Abstract base class for the Red Pitaya (or simulated) backend."""

    num_actuators: int = 2

    def apply_control(self, controls: Sequence[float]) -> None:
        """Apply the given control vector to the actuators."""
        raise NotImplementedError

    def measure_intensity(self, averages: int = 1) -> float:
        """Return the detector intensity averaged over ``averages`` samples."""
        raise NotImplementedError

    def efficiency(self, intensity: float) -> float:
        """Return the coherent synthesis efficiency for the given intensity."""
        raise NotImplementedError

    def reference_intensity(self) -> float:
        """Return the best intensity that has been observed so far."""
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - trivial default
        """Release hardware resources if necessary."""
        return None


class RedPitayaBackend(Backend):
    """Minimal SCPI-based backend for the Red Pitaya STEMlab 125-14."""

    def __init__(
        self,
        host: str,
        port: int = 5000,
        output_channels: Sequence[int] = (1, 2),
        input_channel: int = 1,
        voltage_range: float = 1.0,
        averages: int = 1,
        sample_wait: float = 3e-3,
    ) -> None:
        self.host = host
        self.port = port
        self.output_channels = tuple(output_channels)
        self.input_channel = int(input_channel)
        self.voltage_range = float(voltage_range)
        self.averages = max(1, int(averages))
        self.sample_wait = float(sample_wait)
        self._sock = socket.create_connection((host, port), timeout=1.0)
        self._sock.settimeout(1.0)
        self._buffer = b""
        self._last_reference = 1e-9
        self._initialise_outputs()

    def _initialise_outputs(self) -> None:
        for ch in self.output_channels:
            self._send_command(f"SOUR{ch}:FUNC DC")
            self._send_command(f"SOUR{ch}:VOLT 0")
            self._send_command(f"SOUR{ch}:STATE ON")
        self._send_command("ACQ:RST")
        self._send_command("ACQ:DEC 1")
        self._send_command("ACQ:AVG ON")
        self._send_command("ACQ:START")
        time.sleep(0.01)

    def _send_command(self, command: str) -> None:
        self._sock.sendall((command + "\n").encode("ascii"))

    def _query(self, command: str) -> str:
        self._send_command(command)
        data = b""
        while not data.endswith(b"\n"):
            chunk = self._sock.recv(4096)
            if not chunk:
                raise BackendError("Connection to Red Pitaya closed unexpectedly")
            data += chunk
        return data.decode("ascii").strip()

    def apply_control(self, controls: Sequence[float]) -> None:
        if len(controls) != len(self.output_channels):
            raise ValueError("Control vector dimensionality does not match outputs")
        for value, channel in zip(controls, self.output_channels):
            value = max(-1.0, min(1.0, float(value)))
            value *= self.voltage_range
            self._send_command(f"SOUR{channel}:VOLT {value}")
        # Keep acquisition running so that measurements are ready when requested.
        self._send_command("ACQ:START")

    def measure_intensity(self, averages: int = 1) -> float:
        averages = max(1, int(averages))
        readings: List[float] = []
        for _ in range(averages):
            time.sleep(self.sample_wait)
            self._send_command("ACQ:START")
            self._send_command("ACQ:TRIG NOW")
            time.sleep(self.sample_wait)
            raw = self._query(f"ACQ:SOUR{self.input_channel}:VALUE?")
            try:
                value = float(raw)
            except ValueError as exc:  # pragma: no cover - depends on hardware
                raise BackendError(f"Invalid ADC reading: {raw!r}") from exc
            readings.append(value)
        intensity = statistics.fmean(readings)
        self._last_reference = max(self._last_reference, intensity)
        return intensity

    def efficiency(self, intensity: float) -> float:
        if self._last_reference <= 0:
            return 0.0
        return max(0.0, min(1.0, intensity / self._last_reference))

    def reference_intensity(self) -> float:
        return self._last_reference

    def close(self) -> None:  # pragma: no cover - requires hardware
        try:
            self._send_command("SOUR1:STATE OFF")
            self._send_command("SOUR2:STATE OFF")
        finally:
            self._sock.close()


class SimulatedBackend(Backend):
    """Simple physical model for two-channel coherent synthesis."""

    def __init__(
        self,
        i1: float = 1.0,
        i2: float = 1.0,
        phase_noise: float = 0.02,
        amplitude_noise: float = 0.01,
        drift_per_sample: float = 0.01,
        random_seed: Optional[int] = None,
    ) -> None:
        self.i1 = float(i1)
        self.i2 = float(i2)
        self.phase_noise = float(phase_noise)
        self.amplitude_noise = float(amplitude_noise)
        self.drift_per_sample = float(drift_per_sample)
        self.random = random.Random(random_seed)
        self._current_controls = np.zeros(2)
        self._phase_state = self.random.random() * 2 * math.pi
        self._amp_state = 1.0
        self._max_intensity = self._max_theoretical_intensity()
        self._reference = 1e-9

    def _max_theoretical_intensity(self) -> float:
        return (math.sqrt(self.i1) + math.sqrt(self.i2)) ** 2

    def _update_free_running_channel(self) -> None:
        # Phase performs a random walk.
        self._phase_state += self.random.gauss(0.0, self.drift_per_sample)
        self._phase_state %= 2 * math.pi
        # Amplitude drifts slowly around 1.
        self._amp_state += self.random.gauss(0.0, self.drift_per_sample * 0.1)
        self._amp_state = max(0.2, min(2.0, self._amp_state))

    def apply_control(self, controls: Sequence[float]) -> None:
        if len(controls) != 2:
            raise ValueError("Simulated backend expects two control values")
        self._current_controls = np.asarray(controls, dtype=float)

    def measure_intensity(self, averages: int = 1) -> float:
        averages = max(1, int(averages))
        readings = []
        for _ in range(averages):
            self._update_free_running_channel()
            phase_error = self._phase_state - self._current_controls[0]
            amplitude_scale = max(0.0, self._amp_state + self._current_controls[1])
            i2 = self.i2 * amplitude_scale
            noise_phase = self.random.gauss(0.0, self.phase_noise)
            noise_amp = self.random.gauss(0.0, self.amplitude_noise)
            i1_noisy = self.i1 * (1.0 + noise_amp)
            interference = 2.0 * math.sqrt(abs(i1_noisy) * abs(i2))
            intensity = (
                max(i1_noisy, 0.0)
                + max(i2, 0.0)
                + interference * math.cos(phase_error + noise_phase)
            )
            # Add detector shot-noise like term.
            intensity += self.random.gauss(0.0, 0.01 * self._max_intensity)
            readings.append(max(intensity, 0.0))
        result = float(np.mean(readings))
        self._reference = max(self._reference, result)
        return result

    def efficiency(self, intensity: float) -> float:
        return max(0.0, min(1.0, intensity / self._max_intensity))

    def reference_intensity(self) -> float:
        return self._reference


@dataclass
class ControllerConfig:
    iterations: int = 600
    gain: float = 0.05
    perturbation: float = 0.04
    averages: int = 1
    sample_delay: float = 0.002
    auto_tune: bool = False
    auto_tune_stages: int = 2
    tolerance: float = 1e-4
    settle: int = 30
    report_every: int = 10
    save_path: Optional[str] = None
    save_transfer_function: bool = True
    save_noise_spectrum: bool = True
    random_seed: Optional[int] = None


@dataclass
class LockResult:
    controls: List[Tuple[float, float]] = field(default_factory=list)
    intensities: List[float] = field(default_factory=list)
    efficiencies: List[float] = field(default_factory=list)
    best_intensity: float = 0.0
    best_controls: Tuple[float, float] = (0.0, 0.0)
    timestamp: float = field(default_factory=time.time)


class LivePlotter:
    """Helper for real-time plotting of intensity and efficiency."""

    def __init__(self, max_points: int = 500) -> None:
        if plt is None or FuncAnimation is None:  # pragma: no cover - optional
            raise RuntimeError("matplotlib is required for live plotting")
        self.max_points = max_points
        plt.ion()
        self.fig, (self.ax_intensity, self.ax_efficiency) = plt.subplots(2, 1, sharex=True)
        self.fig.suptitle("SPGD coherent locking")
        self.line_intensity, = self.ax_intensity.plot([], [], label="Detector intensity")
        self.line_efficiency, = self.ax_efficiency.plot([], [], label="Efficiency")
        self.ax_intensity.set_ylabel("Intensity [arb. units]")
        self.ax_efficiency.set_ylabel("Efficiency")
        self.ax_efficiency.set_xlabel("Iteration")
        self.ax_efficiency.set_ylim(0.0, 1.05)
        self.ax_intensity.grid(True)
        self.ax_efficiency.grid(True)
        self.animation = FuncAnimation(self.fig, self._animate, interval=200)
        self._history_iterations: Deque[int] = deque(maxlen=max_points)
        self._history_intensity: Deque[float] = deque(maxlen=max_points)
        self._history_efficiency: Deque[float] = deque(maxlen=max_points)

    def _animate(self, _frame: int):  # pragma: no cover - animation callback
        self.line_intensity.set_data(self._history_iterations, self._history_intensity)
        self.line_efficiency.set_data(self._history_iterations, self._history_efficiency)
        if self._history_iterations:
            xmin = max(0, self._history_iterations[0])
            xmax = self._history_iterations[-1]
            self.ax_intensity.set_xlim(xmin, xmax + 1)
            ymax = max(self._history_intensity) * 1.1
            self.ax_intensity.set_ylim(0.0, max(1e-6, ymax))
        self.ax_intensity.legend(loc="upper left")
        self.ax_efficiency.legend(loc="lower right")
        return self.line_intensity, self.line_efficiency

    def update(self, iteration: int, intensity: float, efficiency: float) -> None:
        self._history_iterations.append(iteration)
        self._history_intensity.append(intensity)
        self._history_efficiency.append(efficiency)
        if plt is not None:  # pragma: no cover - optional update
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()


class SPGDLockController:
    """Implements the SPGD control logic."""

    def __init__(
        self,
        backend: Backend,
        config: ControllerConfig,
        plotter: Optional[LivePlotter] = None,
    ) -> None:
        self.backend = backend
        self.config = config
        self.plotter = plotter
        self.random = random.Random(config.random_seed)

    def _spgd_step(
        self, controls: np.ndarray, perturbation: float, gain: float
    ) -> Tuple[np.ndarray, float, float, float]:
        delta = np.array([self.random.choice((-1.0, 1.0)) for _ in range(len(controls))])
        self.backend.apply_control(controls + perturbation * delta)
        time.sleep(self.config.sample_delay)
        y_plus = self.backend.measure_intensity(self.config.averages)
        self.backend.apply_control(controls - perturbation * delta)
        time.sleep(self.config.sample_delay)
        y_minus = self.backend.measure_intensity(self.config.averages)
        gradient = ((y_plus - y_minus) / (2.0 * perturbation)) * delta
        new_controls = controls + gain * gradient
        self.backend.apply_control(new_controls)
        time.sleep(self.config.sample_delay)
        measurement = self.backend.measure_intensity(self.config.averages)
        efficiency = self.backend.efficiency(measurement)
        return new_controls, measurement, efficiency, float(np.linalg.norm(gradient))

    def auto_tune(self, controls: np.ndarray) -> Tuple[float, float]:
        if not self.config.auto_tune:
            return self.config.gain, self.config.perturbation
        gain = self.config.gain
        perturbation = self.config.perturbation
        for stage in range(max(1, self.config.auto_tune_stages)):
            responses = []
            for _ in range(10):
                _, measurement, _, grad_norm = self._spgd_step(controls, perturbation, gain)
                responses.append((measurement, grad_norm))
            mean_meas = float(np.mean([r[0] for r in responses]))
            mean_grad = float(np.mean([r[1] for r in responses]))
            if mean_grad < self.config.tolerance:
                gain *= 1.5
                perturbation *= 0.7
            elif mean_meas < self.backend.reference_intensity() * 0.8:
                perturbation *= 1.3
            else:
                gain *= 0.9
                perturbation *= 0.9
            gain = max(1e-4, min(1.0, gain))
            perturbation = max(1e-4, min(0.5, perturbation))
        return gain, perturbation

    def run(self) -> LockResult:
        controls = np.zeros(self.backend.num_actuators, dtype=float)
        gain, perturbation = self.auto_tune(controls.copy())
        result = LockResult()
        gradient_history: Deque[float] = deque(maxlen=self.config.settle)
        for iteration in range(1, self.config.iterations + 1):
            controls, intensity, efficiency, grad_norm = self._spgd_step(
                controls, perturbation, gain
            )
            result.controls.append(tuple(float(x) for x in controls))
            result.intensities.append(float(intensity))
            result.efficiencies.append(float(efficiency))
            gradient_history.append(grad_norm)
            if intensity > result.best_intensity:
                result.best_intensity = float(intensity)
                result.best_controls = tuple(float(x) for x in controls)
            if self.plotter is not None:
                self.plotter.update(iteration, intensity, efficiency)
            if iteration % self.config.report_every == 0:
                print(
                    f"Iteration {iteration:5d}: intensity={intensity:8.4f} "
                    f"efficiency={efficiency:6.3f} grad_norm={grad_norm:7.4f}"
                )
            if (
                iteration > self.config.settle
                and len(gradient_history) == gradient_history.maxlen
                and max(gradient_history) < self.config.tolerance
            ):
                print(
                    "Stopping early because the gradient norm has been within the "
                    "tolerance window for the configured settle period."
                )
                break
        self.backend.apply_control(result.best_controls)
        return result


def estimate_transfer_function(
    controls: Sequence[Tuple[float, float]],
    intensities: Sequence[float],
    sample_period: float,
    nfft: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the closed-loop transfer function from logged data."""
    if len(controls) < 2 or len(intensities) != len(controls):
        raise ValueError("Controls and intensity history must have the same length")
    control_array = np.asarray(controls)
    intensity_array = np.asarray(intensities)
    if nfft is None:
        nfft = 1
        while nfft < len(control_array):
            nfft <<= 1
    window = np.hanning(len(control_array))
    control_fft = np.fft.rfft((control_array[:, 0]) * window, n=nfft)
    intensity_fft = np.fft.rfft(intensity_array * window, n=nfft)
    frequency = np.fft.rfftfreq(nfft, d=sample_period)
    eps = 1e-12
    transfer = intensity_fft / np.where(abs(control_fft) < eps, eps, control_fft)
    return frequency, transfer


def estimate_intensity_noise(
    intensities: Sequence[float],
    sample_period: float,
    nfft: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the single-sided PSD of the residual intensity noise."""
    if len(intensities) < 2:
        raise ValueError("At least two intensity samples are required")
    data = np.asarray(intensities)
    if nfft is None:
        nfft = 1
        while nfft < len(data):
            nfft <<= 1
    window = np.hanning(len(data))
    windowed = data * window
    spectrum = np.fft.rfft(windowed, n=nfft)
    psd = (np.abs(spectrum) ** 2) * (2.0 * sample_period / (np.sum(window ** 2)))
    frequency = np.fft.rfftfreq(nfft, d=sample_period)
    return frequency, psd


def save_results(
    result: LockResult,
    transfer: Tuple[np.ndarray, np.ndarray],
    noise: Tuple[np.ndarray, np.ndarray],
    config: ControllerConfig,
    path: Path,
) -> None:
    payload = {
        "timestamp": result.timestamp,
        "controls": result.controls,
        "intensities": result.intensities,
        "efficiencies": result.efficiencies,
        "best_intensity": result.best_intensity,
        "best_controls": result.best_controls,
        "config": dataclasses.asdict(config),
        "transfer_function": {
            "frequency": transfer[0].tolist(),
            "response_real": transfer[1].real.tolist(),
            "response_imag": transfer[1].imag.tolist(),
        },
        "intensity_noise": {
            "frequency": noise[0].tolist(),
            "psd": noise[1].tolist(),
        },
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"Saved results to {path}")


def load_config(path: Optional[str]) -> Dict[str, object]:
    if not path:
        return {}
    with open(path, "r", encoding="utf8") as handle:
        return json.load(handle)


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", help="Hostname or IP address of the Red Pitaya")
    parser.add_argument("--port", type=int, default=5000, help="SCPI port of the device")
    parser.add_argument("--iterations", type=int, help="Number of SPGD iterations")
    parser.add_argument("--gain", type=float, help="SPGD gain")
    parser.add_argument("--perturbation", type=float, help="Perturbation amplitude")
    parser.add_argument("--averages", type=int, help="ADC averages per measurement")
    parser.add_argument("--sample-delay", type=float, help="Delay between control updates")
    parser.add_argument("--auto-tune", action="store_true", help="Enable automatic tuning")
    parser.add_argument(
        "--auto-tune-stages", type=int, help="Number of auto-tuning stages to run"
    )
    parser.add_argument("--tolerance", type=float, help="Gradient-norm tolerance")
    parser.add_argument("--settle", type=int, help="Iterations before convergence check")
    parser.add_argument("--report-every", type=int, help="Report progress every N iterations")
    parser.add_argument("--simulate", action="store_true", help="Run the simulation backend")
    parser.add_argument(
        "--simulation-noise",
        type=float,
        default=0.02,
        help="Standard deviation of the simulated phase noise",
    )
    parser.add_argument(
        "--simulation-drift",
        type=float,
        default=0.01,
        help="Random-walk step size of the simulated free-running channel",
    )
    parser.add_argument("--plot", action="store_true", help="Show live plots during locking")
    parser.add_argument("--config", help="Load default options from a JSON file")
    parser.add_argument("--save", help="Store the run summary in JSON format")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    return parser.parse_args(argv)


def merge_config(args: argparse.Namespace) -> Tuple[ControllerConfig, Dict[str, object]]:
    config_data = load_config(args.config)
    cfg = ControllerConfig()
    for field_ in dataclasses.fields(cfg):
        if field_.name in config_data:
            setattr(cfg, field_.name, config_data[field_.name])
    for key in (
        "iterations",
        "gain",
        "perturbation",
        "averages",
        "sample_delay",
        "auto_tune",
        "auto_tune_stages",
        "tolerance",
        "settle",
        "report_every",
        "save_path",
        "save_transfer_function",
        "save_noise_spectrum",
    ):
        arg_name = key.replace("_", "-")
        if hasattr(args, arg_name):
            value = getattr(args, arg_name)
            if value is not None:
                setattr(cfg, key, value)
    if args.seed is not None:
        cfg.random_seed = args.seed
    backend_params: Dict[str, object] = {}
    if args.simulate:
        backend_params.update(
            phase_noise=args.simulation_noise,
            drift_per_sample=args.simulation_drift,
        )
    else:
        backend_params.update(host=args.host, port=args.port)
        if args.host is None:
            raise SystemExit("--host is required when not running the simulation backend")
    if args.save is not None:
        cfg.save_path = args.save
    return cfg, backend_params


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_arguments(argv)
    cfg, backend_params = merge_config(args)
    backend: Backend
    if args.simulate:
        backend = SimulatedBackend(
            phase_noise=float(backend_params.get("phase_noise", 0.02)),
            drift_per_sample=float(backend_params.get("drift_per_sample", 0.01)),
            random_seed=cfg.random_seed,
        )
        print("Running in simulation mode")
    else:
        backend = RedPitayaBackend(
            host=str(backend_params["host"]),
            port=int(backend_params.get("port", 5000)),
            averages=cfg.averages,
        )
        print(f"Connected to Red Pitaya at {backend_params['host']}:{backend_params.get('port', 5000)}")
    plotter: Optional[LivePlotter] = None
    if args.plot:
        if plt is None:
            raise SystemExit("matplotlib is required for --plot")
        plotter = LivePlotter()
    controller = SPGDLockController(backend, cfg, plotter=plotter)
    try:
        result = controller.run()
        print("Best intensity:", result.best_intensity)
        print("Best controls:", result.best_controls)
        sample_period = max(cfg.sample_delay, 1e-3)
        transfer = estimate_transfer_function(result.controls, result.intensities, sample_period)
        noise = estimate_intensity_noise(result.intensities, sample_period)
        print("Transfer function and intensity noise estimated.")
        if cfg.save_path:
            save_results(result, transfer, noise, cfg, Path(cfg.save_path))
    finally:
        backend.close()
    if plotter is not None and plt is not None:
        print("Close the plot window to exit.")
        plt.ioff()
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
