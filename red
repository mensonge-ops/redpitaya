#!/usr/bin/env python3
"""Perform two-channel coherent locking with SPGD on a Red Pitaya.

The script implements a two-channel simultaneous perturbation stochastic
gradient descent (SPGD) controller that drives two actuators (for example two
phase modulators) through the analog outputs of a Red Pitaya STEMlab 125-14.
The controller maximizes the detected combined intensity that is acquired via
one of the fast ADC channels.  After convergence the script can estimate the
closed-loop transfer function by injecting small sinusoidal perturbations and
analyse the residual intensity noise spectrum.

The file also contains a light-weight simulation backend so the control loop
can be tested without connecting to the hardware by passing ``--simulate``.

Example usage with hardware::

    python examples/red_pitaya_spgd_lock.py --host 192.168.1.100 \
        --iterations 600 --gain 0.08 --perturbation 0.04

Example usage in simulation mode::

    python examples/red_pitaya_spgd_lock.py --simulate --plot

For repeated experiments you can store the desired command-line options in a
JSON file and load them with ``--config path/to/file.json``.  The repository
ships with ``examples/red_pitaya_spgd_lock_config.json`` as a starting point.
You can tweak gain, perturbation and auto-tuning stages there without touching
the code and still override any value from the command line when needed.

To let the script automatically refine the SPGD gain and perturbation amplitudes
use the ``--auto-tune`` flag (optionally together with ``--auto-tune-stages``)::

    python examples/red_pitaya_spgd_lock.py --simulate --auto-tune --auto-tune-stages 3

The script prints a concise textual summary and optionally stores the results
on disk.  It does not require the rest of the :mod:`pychi` package and can be
run independently as long as the dependencies listed in ``requirements.txt``
are installed.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
try:
    import matplotlib  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    matplotlib = None  # type: ignore[assignment]
else:  # pragma: no branch - tiny loop
    for backend in ("QtAgg", "Qt5Agg", "TkAgg"):
        try:
            matplotlib.use(backend)
            break
        except Exception:
            pass

import numpy as np
from scipy import signal

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes describing the different stages of the locking experiment.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SPGDAutoTuneStage:
    """Single stage of an auto-tuning schedule."""

    iterations: int
    gain_scale: float = 1.0
    perturbation_scale: float = 1.0


@dataclass
class SPGDConfig:
    """Configuration parameters for the SPGD controller."""

    iterations: int = 600
    gain: float = 0.08
    perturbation: float = 0.04
    perturbation_decay: float = 0.997
    settling_time: float = 0.005
    metric_average: int = 1024
    control_limits: Tuple[float, float] = (-1.8, 1.8)
    auto_stages: Optional[Sequence[SPGDAutoTuneStage]] = None


@dataclass
class LockingResult:
    """Container storing the raw data collected during the locking stage."""

    control_history: np.ndarray
    metric_history: np.ndarray
    final_control: np.ndarray
    final_metric: float
    stages: Sequence[SPGDAutoTuneStage]
    stage_boundaries: np.ndarray


@dataclass
class TransferFunctionResult:
    """Transfer function estimate obtained after the loop has settled."""

    frequencies: np.ndarray
    magnitude: np.ndarray
    phase: np.ndarray


@dataclass
class NoiseAnalysisResult:
    """Result of the residual intensity noise analysis."""

    time: np.ndarray
    samples: np.ndarray
    frequency: np.ndarray
    psd: np.ndarray


@dataclass(frozen=True)
class ConfigFileDefaults:
    """Container for configuration values loaded from ``--config``."""

    cli_args: Dict[str, Any]
    auto_stages: Optional[Sequence[SPGDAutoTuneStage]] = None


def load_config_file(path: Path) -> ConfigFileDefaults:
    """Load default command-line arguments from a JSON configuration file."""

    path = path.expanduser()
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:  # pragma: no cover - depends on user input
        raise ValueError(f"Configuration file '{path}' does not exist") from exc
    except json.JSONDecodeError as exc:  # pragma: no cover - depends on user input
        raise ValueError(f"Configuration file '{path}' is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Top-level configuration has to be a JSON object")

    raw_auto_stages = data.pop("auto_stages", None)

    allowed_keys = {
        "host",
        "simulate",
        "iterations",
        "gain",
        "perturbation",
        "perturbation_decay",
        "auto_tune",
        "auto_tune_stages",
        "settling_time",
        "metric_average",
        "control_min",
        "control_max",
        "transfer_frequencies",
        "transfer_amplitude",
        "transfer_duration",
        "noise_duration",
        "decimation",
        "metric_channel",
        "log_level",
        "plot",
    }

    unexpected = set(data) - allowed_keys
    if unexpected:
        raise ValueError(f"Unsupported option(s) in configuration file: {sorted(unexpected)}")

    cli_args: Dict[str, Any] = {}
    for key, value in data.items():
        if key == "transfer_frequencies":
            if not isinstance(value, (list, tuple)):
                raise ValueError("'transfer_frequencies' must be a list of numbers")
            cli_args[key] = [float(v) for v in value]
        else:
            cli_args[key] = value

    auto_stages: Optional[List[SPGDAutoTuneStage]] = None
    if raw_auto_stages is not None:
        if not isinstance(raw_auto_stages, list):
            raise ValueError("'auto_stages' must be a list of objects")
        auto_stages = []
        for idx, entry in enumerate(raw_auto_stages, start=1):
            if not isinstance(entry, dict):
                raise ValueError(f"Auto-tune stage {idx} must be a JSON object")
            if "iterations" not in entry:
                raise ValueError(f"Auto-tune stage {idx} is missing the 'iterations' field")
            iterations = int(entry["iterations"])
            if iterations <= 0:
                raise ValueError(f"Auto-tune stage {idx} must have a positive number of iterations")
            gain_scale = float(entry.get("gain_scale", 1.0))
            perturbation_scale = float(entry.get("perturbation_scale", 1.0))
            auto_stages.append(
                SPGDAutoTuneStage(
                    iterations=iterations,
                    gain_scale=gain_scale,
                    perturbation_scale=perturbation_scale,
                )
            )
        if not auto_stages:
            auto_stages = None

    return ConfigFileDefaults(cli_args=cli_args, auto_stages=auto_stages)


def build_auto_tune_schedule(iterations: int, stage_count: int) -> List[SPGDAutoTuneStage]:
    """Create a geometric auto-tuning schedule.

    The helper spreads ``iterations`` across ``stage_count`` stages and
    progressively reduces the perturbation and gain scaling factors to refine
    the lock point.  The last stage always receives any leftover iterations so
    that the sum matches ``iterations``.
    """

    if iterations <= 0:
        raise ValueError("Number of iterations has to be positive for auto-tuning")
    stage_count = max(1, stage_count)
    stage_count = min(stage_count, iterations)
    counts: List[int] = []
    remaining = iterations
    for idx in range(stage_count):
        stages_left = stage_count - idx
        share = max(1, remaining // stages_left)
        counts.append(share)
        remaining -= share
    if remaining:
        counts[-1] += remaining
    gain_scales = np.geomspace(1.0, 0.35, stage_count)
    perturb_scales = np.geomspace(1.0, 0.2, stage_count)
    return [
        SPGDAutoTuneStage(iterations=count, gain_scale=float(g), perturbation_scale=float(p))
        for count, g, p in zip(counts, gain_scales, perturb_scales)
    ]


def wrap_to_pi(values: np.ndarray) -> np.ndarray:
    """Wrap an array of angles to the ``[-pi, pi]`` interval."""

    return (values + np.pi) % (2.0 * np.pi) - np.pi


class LiveIntensityPlot:
    """Helper to stream detector intensity in real time using matplotlib."""

    def __init__(self, ax, sample_rate: float, window: float = 1.0) -> None:
        self.ax = ax
        self.sample_rate = sample_rate
        self.window = window
        self._time = np.empty(0, dtype=float)
        self._samples = np.empty(0, dtype=float)
        (self._line,) = ax.plot([], [], lw=1)
        ax.set_title("Detector intensity (live)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Intensity (a.u.)")

    def update(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        start_time = 0.0 if self._time.size == 0 else self._time[-1] + 1.0 / self.sample_rate
        new_time = start_time + np.arange(samples.size) / self.sample_rate
        self._time = np.concatenate((self._time, new_time))
        self._samples = np.concatenate((self._samples, samples))
        if self.window > 0:
            mask = self._time >= (self._time[-1] - self.window)
        else:
            mask = slice(None)
        time_view = self._time[mask]
        samples_view = self._samples[mask]
        self._line.set_data(time_view, samples_view)
        if time_view.size:
            self.ax.set_xlim(time_view[0], time_view[-1])
            ymin = float(np.min(samples_view))
            ymax = float(np.max(samples_view))
            if ymin == ymax:
                delta = abs(ymin) * 0.1 or 1.0
                ymin -= delta
                ymax += delta
            self.ax.set_ylim(ymin, ymax)
        self.ax.figure.canvas.draw_idle()
        # ``pause`` keeps the UI responsive without blocking the acquisition loop.
        import matplotlib.pyplot as plt  # local import to avoid hard dependency

        plt.pause(0.001)

    def finalize(self) -> None:
        if self._time.size == 0:
            return
        self._line.set_data(self._time, self._samples)
        self.ax.set_xlim(self._time[0], self._time[-1])
        ymin = float(np.min(self._samples))
        ymax = float(np.max(self._samples))
        if ymin == ymax:
            delta = abs(ymin) * 0.1 or 1.0
            ymin -= delta
            ymax += delta
        self.ax.set_ylim(ymin, ymax)

    @property
    def samples(self) -> np.ndarray:
        return self._samples


# ---------------------------------------------------------------------------
# Hardware abstraction layers.
# ---------------------------------------------------------------------------


class LockHardware:
    """Abstract interface used by :class:`LockingController`.

    Only three primitives are required: applying a pair of control signals,
    reading back the detected intensity time series and optional cleanup.
    """

    sample_rate: float

    def set_control(self, control: Sequence[float]) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def read_metric(self, num_samples: int) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def close(self) -> None:
        return


class RedPitayaHardware(LockHardware):
    """Red Pitaya STEMlab 125-14 backend using SCPI commands over TCP/IP.

    The implementation keeps the API deliberately high level and uses ASCII
    transfers which makes it easy to understand and debug.  For high speed
    data taking a dedicated streaming solution is recommended, yet the present
    approach is perfectly adequate for the rather low bandwidth requirements
    of the SPGD controller.
    """

    def __init__(
        self,
        host: str,
        port: int = 5025,
        metric_channel: int = 1,
        decimation: int = 64,
        sample_rate: Optional[float] = None,
        timeout: float = 5.0,
    ) -> None:
        if metric_channel not in (1, 2):
            raise ValueError("metric_channel has to be 1 or 2")
        self.host = host
        self.port = port
        self.metric_channel = metric_channel
        self.decimation = decimation
        # The STEMlab 125-14 has a 125 MSa/s ADC.  Decimation reduces the rate.
        self.sample_rate = sample_rate or 125e6 / decimation
        self.timeout = timeout
        self._socket: Optional[socket.socket] = None
        self._current_control = np.zeros(2, dtype=float)

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> None:
        _LOGGER.info("Connecting to Red Pitaya at %s:%d", self.host, self.port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect((self.host, self.port))
        self._socket = sock
        for idx in (1, 2):
            self._write(f"SOUR{idx}:FUNC ARBITRARY")
            self._write(f"SOUR{idx}:VOLT:OFFS 0")
            self._write(f"SOUR{idx}:STATE ON")
        self._write("ACQ:RESET")
        self._write(f"ACQ:DEC {self.decimation}")
        self._write("ACQ:TRIG:DLY 0")
        self._write("ACQ:TRIG:LEV 0.0")
        self._write("ACQ:TRIG:EXT POS")
        self._write("ACQ:TRIG:SOUR NOW")
        self._write("ACQ:AVG OFF")

    def close(self) -> None:
        if self._socket is None:
            return
        _LOGGER.info("Closing Red Pitaya connection")
        for idx in (1, 2):
            self._write(f"SOUR{idx}:STATE OFF")
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._socket.close()
        self._socket = None

    # -- SCPI helpers ------------------------------------------------------
    def _ensure_socket(self) -> socket.socket:
        if self._socket is None:
            raise RuntimeError("Red Pitaya connection not initialised")
        return self._socket

    def _write(self, command: str) -> None:
        sock = self._ensure_socket()
        if not command.endswith("\n"):
            command += "\n"
        sock.sendall(command.encode("ascii"))

    def _query(self, command: str) -> str:
        sock = self._ensure_socket()
        self._write(command)
        chunks: List[bytes] = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            if chunk.endswith(b"\n"):
                break
        return b"".join(chunks).decode("ascii").strip()

    # -- LockHardware API --------------------------------------------------
    def set_control(self, control: Sequence[float]) -> None:
        control = np.asarray(control, dtype=float)
        if control.shape != (2,):
            raise ValueError("Control vector must have shape (2,)")
        self._current_control = control.copy()
        for idx, value in enumerate(control, start=1):
            self._write(f"SOUR{idx}:VOLT:OFFS {value:.6f}")

    def read_metric(self, num_samples: int) -> np.ndarray:
        self._write("ACQ:STOP")
        self._write("ACQ:RESET")
        self._write("ACQ:START")
        time.sleep(max(num_samples / self.sample_rate, 1e-4))
        raw = self._query(f"ACQ:SOUR{self.metric_channel}:DATA? {num_samples}")
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        elif raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        values = [float(x) for x in raw.split(",") if x.strip()]
        return np.asarray(values[:num_samples], dtype=float)


class SimulatedHardware(LockHardware):
    """Noise-injected simulation of a two-arm coherent synthesiser.

    The simulation models two equal amplitude beams with a slowly drifting
    relative phase.  The control inputs are interpreted as phase corrections
    applied by two actuators.  The detected intensity corresponds to the total
    intensity of the coherent sum.
    """

    def __init__(
        self,
        sample_rate: float = 10_000.0,
        noise_std: float = 0.01,
        drift_rate: float = 0.2,
        amplitude: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.noise_std = noise_std
        self.drift_rate = drift_rate
        self.amplitude = amplitude
        self.rng = rng or np.random.default_rng(1234)
        self._control = np.zeros(2, dtype=float)
        self._phase_bias = float(self.rng.uniform(-math.pi, math.pi))

    def set_control(self, control: Sequence[float]) -> None:
        control = np.asarray(control, dtype=float)
        if control.shape != (2,):
            raise ValueError("Control vector must have shape (2,)")
        self._control = control.copy()

    def read_metric(self, num_samples: int) -> np.ndarray:
        increments = self.drift_rate * np.arange(num_samples, dtype=float) / self.sample_rate
        phase_drift = wrap_to_pi(self._phase_bias + increments)
        field = self.amplitude * np.exp(1j * (phase_drift + self._control[0]))
        field += self.amplitude * np.exp(1j * self._control[1])
        intensity = np.abs(field) ** 2
        noise = self.rng.normal(scale=self.noise_std, size=num_samples)
        total_advance = self.drift_rate * num_samples / self.sample_rate
        self._phase_bias = float(wrap_to_pi(self._phase_bias + total_advance))
        return intensity + noise


# ---------------------------------------------------------------------------
# Locking controller implementing the SPGD algorithm.
# ---------------------------------------------------------------------------


class LockingController:
    """SPGD controller orchestrating the locking and analysis procedure."""

    def __init__(
        self,
        hardware: LockHardware,
        config: SPGDConfig,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.hardware = hardware
        self.config = config
        self.rng = rng or np.random.default_rng()
        self._current_control = np.zeros(2, dtype=float)

    # -- utility -----------------------------------------------------------
    def _measure_metric(self) -> float:
        samples = self.hardware.read_metric(self.config.metric_average)
        return float(np.mean(samples))

    def _evaluate(self, control: np.ndarray) -> float:
        self.hardware.set_control(control)
        time.sleep(self.config.settling_time)
        return self._measure_metric()

    # -- main algorithms ---------------------------------------------------
    def run_lock(self) -> LockingResult:
        config = self.config
        control = self._current_control.copy()
        metric_history: List[float] = []
        control_history: List[np.ndarray] = []

        schedule = list(config.auto_stages or [])
        if not schedule:
            schedule = [SPGDAutoTuneStage(iterations=config.iterations)]
        total_iterations = sum(stage.iterations for stage in schedule)
        if total_iterations <= 0:
            raise ValueError("Total number of iterations must be positive")

        best_metric = -np.inf
        best_control = control.copy()
        completed_iterations = 0
        stage_boundaries: List[int] = []

        for stage_index, stage in enumerate(schedule, start=1):
            if stage.iterations <= 0:
                continue
            gain = config.gain * stage.gain_scale
            perturbation = config.perturbation * stage.perturbation_scale
            _LOGGER.info(
                "Stage %d/%d: %d iterations (gain=%.4f, perturbation=%.4f)",
                stage_index,
                len(schedule),
                stage.iterations,
                gain,
                perturbation,
            )
            for local_iter in range(stage.iterations):
                direction = self.rng.choice([-1.0, 1.0], size=2)
                delta = perturbation * direction
                metric_plus = self._evaluate(control + delta)
                metric_minus = self._evaluate(control - delta)
                gradient = (metric_plus - metric_minus) / (2.0 * delta)
                control = control + gain * gradient
                control = np.clip(control, *config.control_limits)
                nominal_metric = self._evaluate(control)

                metric_history.append(nominal_metric)
                control_history.append(control.copy())

                if nominal_metric > best_metric:
                    best_metric = nominal_metric
                    best_control = control.copy()

                perturbation *= config.perturbation_decay
                completed_iterations += 1
                _LOGGER.debug(
                    "Stage %d iter %04d/%04d metric=% .5f control=%s perturb=%.5f gain=%.5f",
                    stage_index,
                    local_iter + 1,
                    stage.iterations,
                    nominal_metric,
                    np.array2string(control, precision=4),
                    perturbation,
                    gain,
                )
            stage_boundaries.append(completed_iterations)

        self._current_control = best_control.copy()
        self.hardware.set_control(best_control)
        final_metric = self._measure_metric()
        metric_history.append(final_metric)
        control_history.append(best_control.copy())

        return LockingResult(
            control_history=np.asarray(control_history),
            metric_history=np.asarray(metric_history),
            final_control=best_control,
            final_metric=final_metric,
            stages=schedule,
            stage_boundaries=np.asarray(stage_boundaries, dtype=int),
        )

    def measure_transfer_function(
        self,
        frequencies: Sequence[float],
        amplitude: float = 0.01,
        duration: float = 1.0,
        drive_vector: Optional[Sequence[float]] = None,
    ) -> TransferFunctionResult:
        if drive_vector is None:
            drive_vector = (1.0, -1.0)
        drive_vector = np.asarray(drive_vector, dtype=float)
        base_control = self._current_control.copy()
        fs = self.hardware.sample_rate
        n_samples = max(int(duration * fs), 1)
        time_step = 1.0 / fs

        magnitudes: List[float] = []
        phases: List[float] = []

        for frequency in frequencies:
            t = np.arange(n_samples) * time_step
            inputs = []
            outputs = []
            for sin_arg in 2 * np.pi * frequency * t:
                perturb = amplitude * math.sin(sin_arg)
                command = base_control + drive_vector * perturb
                self.hardware.set_control(command)
                time.sleep(time_step)
                measurement = self._measure_metric()
                inputs.append(perturb)
                outputs.append(measurement)
            inputs = np.asarray(inputs)
            outputs = np.asarray(outputs)
            outputs -= np.mean(outputs)
            inputs -= np.mean(inputs)
            freqs, pxy = signal.csd(inputs, outputs, fs=fs, nperseg=max(len(inputs) // 8, 8))
            _, pxx = signal.welch(inputs, fs=fs, nperseg=max(len(inputs) // 8, 8))
            transfer = pxy / pxx
            idx = int(np.argmin(np.abs(freqs - frequency)))
            magnitudes.append(np.abs(transfer[idx]))
            phases.append(np.angle(transfer[idx]))
            self.hardware.set_control(base_control)

        self.hardware.set_control(base_control)
        return TransferFunctionResult(
            frequencies=np.asarray(frequencies, dtype=float),
            magnitude=np.asarray(magnitudes, dtype=float),
            phase=np.asarray(phases, dtype=float),
        )

    def analyse_noise(
        self,
        duration: float = 2.0,
        live_callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> NoiseAnalysisResult:
        fs = self.hardware.sample_rate
        total_samples = int(max(duration * fs, 0))
        if total_samples == 0:
            return NoiseAnalysisResult(
                time=np.empty(0, dtype=float),
                samples=np.empty(0, dtype=float),
                frequency=np.empty(0, dtype=float),
                psd=np.empty(0, dtype=float),
            )

        collected: List[np.ndarray] = []
        collected_samples = 0
        # Use modest chunks to keep the UI responsive during live plotting.
        chunk_size = int(min(max(fs * 0.02, 256), 65536))

        while collected_samples < total_samples:
            remaining = total_samples - collected_samples
            current_chunk = int(min(chunk_size, remaining))
            samples = self.hardware.read_metric(current_chunk)
            collected.append(samples)
            collected_samples += samples.size
            if live_callback is not None:
                live_callback(samples)

        samples = np.concatenate(collected) if collected else np.empty(0, dtype=float)
        time_axis = np.arange(samples.size) / fs
        demeaned = samples - np.mean(samples) if samples.size else samples
        if demeaned.size:
            nperseg = min(max(demeaned.size // 8, 8), demeaned.size)
            freq, psd = signal.welch(demeaned, fs=fs, nperseg=nperseg)
        else:
            freq = np.empty(0, dtype=float)
            psd = np.empty(0, dtype=float)
        return NoiseAnalysisResult(time=time_axis, samples=samples, frequency=freq, psd=psd)


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        help="Load default arguments from a JSON file so they can be edited without touching the CLI",
    )
    hardware_group = parser.add_mutually_exclusive_group()
    hardware_group.add_argument("--host", type=str, help="Hostname or IP of the Red Pitaya")
    hardware_group.add_argument(
        "--simulate",
        action="store_true",
        help="Run the built-in simulation instead of talking to the hardware",
    )
    parser.add_argument("--iterations", type=int, help="Number of SPGD iterations")
    parser.add_argument("--gain", type=float, help="SPGD integrator gain")
    parser.add_argument("--perturbation", type=float, help="Initial perturbation amplitude")
    parser.add_argument("--perturbation-decay", type=float, help="Per-iteration perturbation decay factor")
    parser.add_argument(
        "--auto-tune",
        dest="auto_tune",
        action="store_true",
        help="Enable a staged schedule that automatically refines gain and perturbation amplitudes",
    )
    parser.add_argument(
        "--no-auto-tune",
        dest="auto_tune",
        action="store_false",
        help="Disable automatic tuning even if enabled via configuration file",
    )
    parser.add_argument(
        "--auto-tune-stages",
        type=int,
        help="Number of refinement stages to use when --auto-tune is active",
    )
    parser.add_argument("--settling-time", type=float, help="Settling time between control updates in seconds")
    parser.add_argument("--metric-average", type=int, help="Number of ADC samples used per metric evaluation")
    parser.add_argument("--control-min", type=float, help="Lower clamp for the control voltages")
    parser.add_argument("--control-max", type=float, help="Upper clamp for the control voltages")
    parser.add_argument(
        "--transfer-frequencies",
        type=float,
        nargs="*",
        help="Frequencies (Hz) at which to estimate the closed-loop transfer function",
    )
    parser.add_argument("--transfer-amplitude", type=float, help="Amplitude of the sinusoidal drive used for the transfer function")
    parser.add_argument("--transfer-duration", type=float, help="Duration per frequency point for the transfer function measurement")
    parser.add_argument("--noise-duration", type=float, help="Duration of the residual intensity noise acquisition")
    parser.add_argument("--decimation", type=int, help="Red Pitaya ADC decimation factor")
    parser.add_argument("--metric-channel", type=int, choices=(1, 2), help="ADC channel carrying the coherent sum detector")
    parser.add_argument("--log-level", type=str, help="Logging level")
    parser.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        help="Plot the results using matplotlib",
    )
    parser.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Skip plotting even if it is enabled via configuration file",
    )
    parser.set_defaults(
        simulate=False,
        iterations=SPGDConfig.iterations,
        gain=SPGDConfig.gain,
        perturbation=SPGDConfig.perturbation,
        perturbation_decay=SPGDConfig.perturbation_decay,
        auto_tune=False,
        auto_tune_stages=3,
        settling_time=SPGDConfig.settling_time,
        metric_average=SPGDConfig.metric_average,
        control_min=SPGDConfig.control_limits[0],
        control_max=SPGDConfig.control_limits[1],
        transfer_frequencies=[5.0, 10.0, 20.0],
        transfer_amplitude=0.01,
        transfer_duration=1.0,
        noise_duration=2.0,
        decimation=64,
        metric_channel=1,
        log_level="INFO",
        plot=False,
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    argv_list = list(argv) if argv is not None else sys.argv[1:]

    config_defaults = ConfigFileDefaults(cli_args={})
    if argv_list:
        config_parser = argparse.ArgumentParser(add_help=False)
        config_parser.add_argument("--config", type=str)
        preliminary_args, _ = config_parser.parse_known_args(argv_list)
        if preliminary_args.config:
            try:
                config_defaults = load_config_file(Path(preliminary_args.config))
            except ValueError as exc:
                raise SystemExit(str(exc))

    parser = build_argument_parser()
    if config_defaults.cli_args:
        parser.set_defaults(**config_defaults.cli_args)
    args = parser.parse_args(argv_list)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    simulate_flag_cli = any(arg == "--simulate" for arg in argv_list)
    host_flag_cli = any(arg == "--host" for arg in argv_list)

    if host_flag_cli:
        if simulate_flag_cli:
            parser.error("Cannot combine --host with --simulate")
        use_simulation = False
    elif simulate_flag_cli:
        use_simulation = True
    elif args.host:
        use_simulation = False
    elif args.simulate:
        use_simulation = True
    else:
        parser.error("Either --simulate or --host must be provided (directly or via --config)")

    config_auto_stages = (
        list(config_defaults.auto_stages) if config_defaults.auto_stages is not None else None
    )
    auto_tune_disabled_cli = any(arg == "--no-auto-tune" for arg in argv_list)
    auto_tune_requested_cli = any(arg == "--auto-tune" for arg in argv_list)
    auto_stages: Optional[List[SPGDAutoTuneStage]]
    auto_schedule_source: Optional[str]
    if (
        config_auto_stages is not None
        and not auto_tune_disabled_cli
        and not auto_tune_requested_cli
    ):
        auto_stages = config_auto_stages
        auto_schedule_source = "config"
    elif args.auto_tune:
        auto_stages = build_auto_tune_schedule(args.iterations, args.auto_tune_stages)
        auto_schedule_source = "cli"
    else:
        auto_stages = None
        auto_schedule_source = None

    if auto_stages is not None:
        iterations_for_config = sum(stage.iterations for stage in auto_stages)
    else:
        iterations_for_config = args.iterations

    config = SPGDConfig(
        iterations=iterations_for_config,
        gain=args.gain,
        perturbation=args.perturbation,
        perturbation_decay=args.perturbation_decay,
        settling_time=args.settling_time,
        metric_average=args.metric_average,
        control_limits=(args.control_min, args.control_max),
        auto_stages=auto_stages,
    )

    if use_simulation:
        hardware: LockHardware = SimulatedHardware()
    else:
        hardware = RedPitayaHardware(
            host=args.host,
            metric_channel=args.metric_channel,
            decimation=args.decimation,
        )
        hardware.connect()

    controller = LockingController(hardware=hardware, config=config)

    fig = None
    plt = None
    live_intensity: Optional[LiveIntensityPlot] = None
    metric_ax = transfer_ax = noise_ax = tf_phase_ax = None

    if args.plot:
        try:
            import matplotlib.pyplot as mpl_plt
        except ImportError:  # pragma: no cover - plotting is optional
            _LOGGER.error("matplotlib is required for plotting. Install it or omit --plot.")
        else:
            plt = mpl_plt
            plt.ion()
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
            live_window = max(min(args.noise_duration, 5.0), 0.5)
            live_intensity = LiveIntensityPlot(axes[0, 0], hardware.sample_rate, window=live_window)
            metric_ax = axes[0, 1]
            metric_ax.set_title("Metric during locking")
            metric_ax.set_xlabel("Iteration")
            metric_ax.set_ylabel("Metric (a.u.)")

            transfer_ax = axes[1, 0]
            transfer_ax.set_title("Transfer function magnitude")
            transfer_ax.set_xlabel("Frequency (Hz)")
            transfer_ax.set_ylabel("Magnitude (dB)")
            tf_phase_ax = transfer_ax.twinx()
            tf_phase_ax.set_ylabel("Phase (rad)")

            noise_ax = axes[1, 1]
            noise_ax.set_title("Residual intensity noise")
            noise_ax.set_xlabel("Frequency (Hz)")
            noise_ax.set_ylabel("PSD (a.u./Hz)")

    schedule_for_log = list(config.auto_stages or [SPGDAutoTuneStage(iterations=config.iterations)])
    total_iterations = sum(stage.iterations for stage in schedule_for_log)

    if auto_stages is not None:
        if auto_schedule_source == "config":
            _LOGGER.info(
                "Auto-tune schedule loaded from %s: %d stage(s), %d total iterations",
                args.config or "configuration file",
                len(schedule_for_log),
                total_iterations,
            )
        else:
            _LOGGER.info(
                "Auto-tune enabled: %d stage(s) covering %d iterations",
                len(schedule_for_log),
                total_iterations,
            )
        for idx, stage in enumerate(schedule_for_log, start=1):
            _LOGGER.info(
                "  Stage %d -> %d iterations, gain=%.4f, perturbation=%.4f",
                idx,
                stage.iterations,
                config.gain * stage.gain_scale,
                config.perturbation * stage.perturbation_scale,
            )
    else:
        _LOGGER.info("Starting SPGD locking with %d iterations", total_iterations)

    try:
        lock_result = controller.run_lock()
        _LOGGER.info("Lock finished. Final metric: %.6f", lock_result.final_metric)

        if metric_ax is not None:
            metric_ax.plot(lock_result.metric_history, label="Metric")
            if auto_stages is not None and lock_result.stage_boundaries.size > 1:
                for boundary in lock_result.stage_boundaries[:-1]:
                    metric_ax.axvline(boundary, color="0.7", linestyle="--", linewidth=1.0)
            metric_ax.legend(loc="best")

        if args.transfer_frequencies:
            _LOGGER.info("Measuring closed-loop transfer function")
            tf_result = controller.measure_transfer_function(
                args.transfer_frequencies,
                amplitude=args.transfer_amplitude,
                duration=args.transfer_duration,
            )
        else:
            tf_result = None

        _LOGGER.info("Analysing residual intensity noise")
        noise_result = controller.analyse_noise(
            duration=args.noise_duration,
            live_callback=live_intensity.update if live_intensity is not None else None,
        )
        if live_intensity is not None:
            live_intensity.finalize()

    finally:
        hardware.close()

    # Print summary -------------------------------------------------------
    print("\n===== SPGD Lock Summary =====")
    print(f"Final control voltages: {lock_result.final_control}")
    print(f"Final metric value:     {lock_result.final_metric:.6f}")
    if auto_stages is not None:
        print("\nAuto-tune schedule:")
        for idx, stage in enumerate(lock_result.stages, start=1):
            stage_gain = config.gain * stage.gain_scale
            stage_perturb = config.perturbation * stage.perturbation_scale
            print(
                f"  Stage {idx}: {stage.iterations} iterations -> gain={stage_gain:.5f}, perturbation={stage_perturb:.5f}"
            )
    if tf_result is not None:
        print("\nClosed-loop transfer function (magnitude |H|, phase in rad):")
        for freq, mag, phase in zip(tf_result.frequencies, tf_result.magnitude, tf_result.phase):
            print(f"  {freq:8.2f} Hz -> |H| = {mag:8.4f}, arg(H) = {phase:8.4f}")
    print("\nResidual intensity noise (Welch PSD):")
    if noise_result.frequency.size:
        print(
            f"  {len(noise_result.frequency)} frequency bins spanning 0..{noise_result.frequency[-1]:.1f} Hz"
        )
    else:
        print("  Insufficient samples to estimate PSD")

    if plt is not None:
        if transfer_ax is not None and tf_result is not None and tf_phase_ax is not None:
            transfer_ax.cla()
            transfer_ax.set_title("Transfer function magnitude")
            transfer_ax.set_xlabel("Frequency (Hz)")
            transfer_ax.set_ylabel("Magnitude (dB)")
            transfer_ax.semilogx(tf_result.frequencies, 20 * np.log10(tf_result.magnitude), label="|H|")
            transfer_ax.legend(loc="best")

            tf_phase_ax.cla()
            tf_phase_ax.set_ylabel("Phase (rad)")
            tf_phase_ax.semilogx(tf_result.frequencies, np.unwrap(tf_result.phase), "C1--", label="arg(H)")
            tf_phase_ax.legend(loc="lower left")
        elif transfer_ax is not None:
            transfer_ax.cla()
            transfer_ax.set_title("Transfer function magnitude")
            transfer_ax.set_xlabel("Frequency (Hz)")
            transfer_ax.set_ylabel("Magnitude (dB)")
            transfer_ax.text(0.5, 0.5, "Transfer measurement skipped", ha="center", va="center")

        if noise_ax is not None:
            noise_ax.cla()
            noise_ax.set_title("Residual intensity noise")
            noise_ax.set_xlabel("Frequency (Hz)")
            noise_ax.set_ylabel("PSD (a.u./Hz)")
            noise_ax.semilogy(noise_result.frequency, noise_result.psd)

        if fig is not None:
            plt.ioff()
            plt.show()


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
