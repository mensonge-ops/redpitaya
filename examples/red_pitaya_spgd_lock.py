#!/usr/bin/env python3
"""Perform two-channel coherent locking with SPGD on a Red Pitaya.

This module contains a feature complete yet hardware agnostic implementation of
simultaneous perturbation stochastic gradient descent (SPGD) that can control
up to two analogue outputs of a Red Pitaya STEMlab 125-14.  The controller can
run directly on the hardware when the :mod:`redpitaya_scpi` package is
available or it can operate on a built-in physics inspired simulation backend
that mimics the behaviour of two phase modulators interfering on a photo
detector.

The code intentionally mirrors the structure of the reference implementation
that previously lived in a different repository.  The main entry point is the
:func:`main` function which exposes a command line interface.  The interface
supports loading JSON configuration files, automatic parameter refinement and
the generation of a concise textual summary of a locking experiment.

Example usage with hardware::

    python examples/red_pitaya_spgd_lock.py --host 192.168.1.100 \
        --iterations 600 --gain 0.03 --perturbation 0.02

Example usage in simulation mode::

    python examples/red_pitaya_spgd_lock.py --simulate --plot

The script prints a concise textual summary and optionally stores the results
on disk.  It does not require the rest of the :mod:`pychi` package and can be
run independently as long as the dependencies listed in ``requirements.txt``
are installed.
"""
from __future__ import annotations

import argparse
import cmath
import json
import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:  # Optional dependency used for hardware access.
    import redpitaya_scpi  # type: ignore
except Exception:  # pragma: no cover - the package is rarely available in CI.
    redpitaya_scpi = None


Number = float


# ---------------------------------------------------------------------------
# Configuration handling
# ---------------------------------------------------------------------------


def _load_json_config(path: Optional[Path]) -> Dict[str, Number]:
    """Load a JSON configuration file.

    Parameters
    ----------
    path:
        Path to the JSON file.  ``None`` skips loading and returns an empty
        dictionary.
    """

    if path is None:
        return {}

    with Path(path).expanduser().open("r", encoding="utf8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):  # pragma: no cover - sanity check
        raise ValueError("Configuration file must contain a JSON object")
    return data


# ---------------------------------------------------------------------------
# Backend definitions
# ---------------------------------------------------------------------------


class AbstractBackend(ABC):
    """Common interface shared by simulated and hardware backends."""

    num_channels: int = 2

    def __init__(self) -> None:
        self._current_drive = [0.0] * self.num_channels

    # ------------------------------------------------------------------
    # Interface expected by the controller
    # ------------------------------------------------------------------
    def measure(self, drive: Sequence[Number]) -> float:
        """Apply ``drive`` to the actuators and return the measured intensity."""

        drive_array = [float(value) for value in drive]
        if len(drive_array) != self.num_channels:  # pragma: no cover - guard
            raise ValueError(
                f"Expected drive vector of length {self.num_channels}, "
                f"received {len(drive_array)}."
            )
        self._current_drive = drive_array
        return float(self._measure_intensity(drive_array))

    @abstractmethod
    def _measure_intensity(self, drive: Sequence[Number]) -> float:
        """Backend specific intensity measurement implementation."""

    # Optional helper hooks -------------------------------------------------
    def clone(self) -> "AbstractBackend":
        """Return a deep copy of the backend if possible.

        Simulation backends override this method which allows the auto tuner to
        run without altering the state of the main controller.  Hardware
        backends return ``self`` because cloning them is not meaningful.
        """

        return self

    def shutdown(self) -> None:
        """Release any resources held by the backend."""

    def __enter__(self) -> "AbstractBackend":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.shutdown()


class SimulatedInterferometerBackend(AbstractBackend):
    """Simulation backend that mimics two phase modulators interfering."""

    def __init__(
        self,
        noise: float = 0.002,
        contrast: float = 0.85,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._rng = random.Random(seed)
        self._reference_field = 1.0 + 0.2j
        self._modulator_couplings = [1.0 * contrast, 0.8 * contrast]
        self._intrinsic_phases = [self._rng.uniform(-math.pi, math.pi) for _ in range(2)]
        self._noise = noise

    def _measure_intensity(self, drive: Sequence[Number]) -> float:
        optical_field = complex(self._reference_field)
        for coupling, phase, command in zip(
            self._modulator_couplings, self._intrinsic_phases, drive
        ):
            optical_field += coupling * cmath.exp(1j * (phase + command))
        intensity = abs(optical_field) ** 2
        noise = self._rng.gauss(0.0, self._noise)
        return float(intensity + noise)

    def clone(self) -> "SimulatedInterferometerBackend":
        seed = self._rng.randint(0, 2**31 - 1)
        clone = SimulatedInterferometerBackend(noise=self._noise, seed=seed)
        clone._reference_field = self._reference_field
        clone._modulator_couplings = list(self._modulator_couplings)
        clone._intrinsic_phases = list(self._intrinsic_phases)
        clone._current_drive = list(self._current_drive)
        return clone


class RedPitayaHardwareBackend(AbstractBackend):
    """Backend that talks to the Red Pitaya via the SCPI interface."""

    def __init__(self, host: str, port: int = 5000) -> None:
        if redpitaya_scpi is None:
            raise RuntimeError(
                "Hardware support requires the 'redpitaya_scpi' package. "
                "Install it with `pip install redpitaya_scpi`."
            )
        super().__init__()
        self._client = redpitaya_scpi.scpi(host, port=port)
        self._configure_outputs()

    def _configure_outputs(self) -> None:
        # We only need a small subset of the Red Pitaya configuration for this
        # demonstration: set both signal generators to DC mode and ensure the
        # fast ADC is armed.  The SCPI command set is well documented by
        # Red Pitaya.  We guard calls because not every firmware revision uses
        # the same syntax, hence the try/except blocks that fall back to
        # alternative commands when available.
        try:
            self._client.tx_txt("OUTPUT1:STATE ON")
            self._client.tx_txt("OUTPUT2:STATE ON")
        except Exception:  # pragma: no cover - depends on firmware
            self._client.tx_txt("OUTPUT1 ON")
            self._client.tx_txt("OUTPUT2 ON")
        self._client.tx_txt("SOUR1:FUNC DC")
        self._client.tx_txt("SOUR2:FUNC DC")
        self._client.tx_txt("SOUR1:VOLT 0")
        self._client.tx_txt("SOUR2:VOLT 0")
        self._client.tx_txt("ACQ:START")
        time.sleep(0.05)

    def _measure_intensity(self, drive: Sequence[Number]) -> float:
        self._client.tx_txt(f"SOUR1:VOLT {drive[0]:.6f}")
        self._client.tx_txt(f"SOUR2:VOLT {drive[1]:.6f}")
        time.sleep(0.001)
        self._client.tx_txt("ACQ:TRIG NOW")
        time.sleep(0.001)
        self._client.tx_txt("ACQ:DATA? CH1")
        raw = self._client.rx_txt()
        cleaned = raw.strip().strip("{}")
        if not cleaned:
            return 0.0
        parts = [float(value) for value in cleaned.split(",") if value.strip()]
        if not parts:
            return 0.0
        tail = parts[-64:] if len(parts) >= 64 else parts
        return sum(tail) / len(tail)

    def shutdown(self) -> None:
        try:
            self._client.tx_txt("OUTPUT1:STATE OFF")
            self._client.tx_txt("OUTPUT2:STATE OFF")
        finally:
            self._client.close()


# ---------------------------------------------------------------------------
# Controller implementation
# ---------------------------------------------------------------------------


@dataclass
class ControllerParameters:
    gain: float
    perturbation: float
    iterations: int
    auto_tune: bool = False
    auto_tune_stages: int = 1
    log_interval: int = 20
    settling_time: float = 0.0
    noise_duration: float = 0.0


@dataclass
class ControllerResult:
    drive_history: List[List[float]]
    intensity_history: List[float]
    gradients: List[List[float]]
    locking_efficiency: List[float] = field(default_factory=list)
    transmission_sweeps: List[Tuple[int, List[float], List[float]]] = field(
        default_factory=list
    )
    noise_spectrum: Tuple[List[float], List[float]] = field(
        default_factory=lambda: ([], [])
    )

    def summary(self) -> str:
        if not self.intensity_history:
            return "No data collected."
        peak = max(self.intensity_history)
        avg = sum(self.intensity_history[-10:]) / min(10, len(self.intensity_history))
        return (
            f"Collected {len(self.intensity_history)} samples. "
            f"Peak intensity {peak:.4f}, running average {avg:.4f}."
        )


def _compute_locking_efficiency(samples: Sequence[float]) -> List[float]:
    efficiencies: List[float] = []
    if not samples:
        return efficiencies

    running_sum = 0.0
    running_max = float("-inf")
    for index, value in enumerate(samples, 1):
        running_sum += value
        if value > running_max:
            running_max = value
        average = running_sum / index
        efficiency = average / running_max if running_max > 0 else 0.0
        efficiencies.append(efficiency)
    return efficiencies


def _estimate_noise_spectrum(samples: Sequence[float]) -> Tuple[List[float], List[float]]:
    if len(samples) < 4:
        return [], []
    try:  # pragma: no cover - optional dependency for plotting diagnostics
        import numpy as np
    except Exception:
        return [], []

    array = np.asarray(samples, dtype=float)
    array = array - array.mean()
    if not array.any():
        return [], []
    spectrum = np.abs(np.fft.rfft(array)) ** 2
    freqs = np.fft.rfftfreq(array.size, d=1.0)
    return freqs.tolist(), spectrum.tolist()


class SPGDController:
    def __init__(self, backend: AbstractBackend, params: ControllerParameters) -> None:
        self.backend = backend
        self.params = params
        self._drive = [0.0] * backend.num_channels
        self._rng = random.Random()

    # ------------------------------------------------------------------
    def run(self) -> ControllerResult:
        params = self.params
        gain = params.gain
        perturb = params.perturbation

        if params.auto_tune:
            gain, perturb = self._auto_tune(
                stages=max(1, params.auto_tune_stages),
                gain=gain,
                perturb=perturb,
            )

        drive_history: List[List[float]] = []
        intensity_history: List[float] = []
        gradients: List[List[float]] = []

        for step in range(params.iterations):
            gradient, intensity = self._spgd_step(gain, perturb)
            for index, component in enumerate(gradient):
                self._drive[index] += gain * component
            drive_history.append(list(self._drive))
            intensity_history.append(intensity)
            gradients.append(gradient)

            if params.log_interval and (step + 1) % params.log_interval == 0:
                print(
                    f"Iteration {step + 1:4d} | intensity {intensity:.4f} | "
                    f"drive {self._drive}"
                )

        locking_efficiency = _compute_locking_efficiency(intensity_history)
        transmission = self._characterise_transmission()
        noise_spectrum = _estimate_noise_spectrum(intensity_history)

        return ControllerResult(
            drive_history,
            intensity_history,
            gradients,
            locking_efficiency,
            transmission,
            noise_spectrum,
        )

    # ------------------------------------------------------------------
    def _spgd_step(self, gain: float, perturb: float) -> Tuple[List[float], float]:
        delta = [self._rng.choice([-1.0, 1.0]) for _ in range(self.backend.num_channels)]
        drive_plus = [value + perturb * d for value, d in zip(self._drive, delta)]
        drive_minus = [value - perturb * d for value, d in zip(self._drive, delta)]
        intensity_plus = self._measure(drive_plus)
        intensity_minus = self._measure(drive_minus)
        gradient = [
            ((intensity_plus - intensity_minus) / (2.0 * perturb)) * d for d in delta
        ]
        trial_drive = [value + gain * g for value, g in zip(self._drive, gradient)]
        new_intensity = self._measure(trial_drive)
        return gradient, new_intensity

    # ------------------------------------------------------------------
    def _auto_tune(self, stages: int, gain: float, perturb: float) -> Tuple[float, float]:
        backend = self.backend
        if backend.clone() is backend:
            print("Auto tuning not supported for this backend; using user parameters.")
            return gain, perturb

        best_metric = float("-inf")
        best_gain = gain
        best_perturb = perturb

        for stage in range(stages):
            scale = 0.5 ** stage
            settle = min(self.params.settling_time * scale, 0.01)
            noise = min(self.params.noise_duration * scale, 0.05)
            measurement_kwargs = {
                "settling_time": settle,
                "noise_duration": noise,
            }
            scaling = [0.5 + 0.25 * i for i in range(5)]
            candidate_gains = [gain * factor for factor in scaling]
            candidate_perts = [perturb * factor for factor in scaling]
            for g in candidate_gains:
                for p in candidate_perts:
                    metric = self._evaluate_candidate(
                        g, p, scale, measurement_kwargs
                    )
                    if metric > best_metric:
                        best_metric, best_gain, best_perturb = metric, g, p
            gain, perturb = best_gain, best_perturb
            print(
                f"Auto tune stage {stage + 1}/{stages}: gain={gain:.4f}, "
                f"perturbation={perturb:.4f}, metric={best_metric:.4f}"
            )

        return best_gain, best_perturb

    def _evaluate_candidate(
        self,
        gain: float,
        perturb: float,
        scale: float,
        measurement_kwargs: Optional[Dict[str, float]] = None,
    ) -> float:
        backend = self.backend.clone()
        drive = list(self._drive)
        measurement_kwargs = measurement_kwargs or {}
        intensity = self._measure(drive, backend=backend, **measurement_kwargs)
        for _ in range(20):
            delta = [self._rng.choice([-1.0, 1.0]) for _ in range(backend.num_channels)]
            plus = [value + perturb * d for value, d in zip(drive, delta)]
            minus = [value - perturb * d for value, d in zip(drive, delta)]
            intensity_plus = self._measure(plus, backend=backend, **measurement_kwargs)
            intensity_minus = self._measure(minus, backend=backend, **measurement_kwargs)
            gradient = [
                ((intensity_plus - intensity_minus) / (2.0 * perturb)) * d for d in delta
            ]
            drive = [value + gain * g * scale for value, g in zip(drive, gradient)]
            intensity = self._measure(drive, backend=backend, **measurement_kwargs)
        return intensity

    # ------------------------------------------------------------------
    def _measure(
        self,
        drive: Sequence[float],
        *,
        backend: Optional[AbstractBackend] = None,
        settling_time: Optional[float] = None,
        noise_duration: Optional[float] = None,
    ) -> float:
        backend = backend or self.backend
        params = self.params
        settling = params.settling_time if settling_time is None else max(0.0, settling_time)
        duration = params.noise_duration if noise_duration is None else max(0.0, noise_duration)

        # Apply the drive first to let the hardware/simulation transition towards
        # the new state.  The initial reading is discarded so that subsequent
        # samples incorporate the optional settling delay.
        backend.measure(drive)

        if settling > 0.0:
            time.sleep(settling)

        if duration <= 0.0:
            return backend.measure(drive)

        # Interpret ``noise_duration`` as an averaging window expressed in seconds
        # and assume roughly twenty samples per second.  This keeps the runtime
        # manageable in simulation while still providing additional noise
        # suppression on real hardware where each acquisition already incurs an
        # inherent delay.
        sample_count = max(1, int(round(duration * 20)))
        samples = [backend.measure(drive) for _ in range(sample_count)]
        return sum(samples) / len(samples)

    # ------------------------------------------------------------------
    def _characterise_transmission(self) -> List[Tuple[int, List[float], List[float]]]:
        if not self.params.iterations:
            return []

        backend = self.backend.clone()
        base_drive = list(self._drive)
        span = max(self.params.perturbation * 6.0, 0.01)
        if span <= 0.0:
            return []

        points = 41
        sweeps: List[Tuple[int, List[float], List[float]]] = []
        offsets = [
            (-span / 2.0) + (span * index) / (points - 1)
            for index in range(points)
        ]

        for channel in range(self.backend.num_channels):
            axis: List[float] = []
            intensities: List[float] = []
            for offset in offsets:
                drive = list(base_drive)
                drive[channel] = base_drive[channel] + offset
                intensity = self._measure(
                    drive,
                    backend=backend,
                    settling_time=0.0,
                    noise_duration=0.0,
                )
                axis.append(drive[channel])
                intensities.append(intensity)

            sweeps.append((channel, axis, intensities))

        # Restore the base drive on the actual backend if we were unable to clone.
        if backend is self.backend:
            self._measure(
                base_drive,
                backend=self.backend,
                settling_time=0.0,
                noise_duration=0.0,
            )

        return sweeps


# ---------------------------------------------------------------------------
# CLI utilities
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", help="Hostname or IP address of the Red Pitaya")
    parser.add_argument("--port", type=int, default=5000, help="SCPI port")
    parser.add_argument("--simulate", action="store_true", help="Use the simulation backend")
    parser.add_argument("--iterations", type=int, default=400, help="Number of SPGD iterations")
    parser.add_argument("--gain", type=float, default=0.03, help="SPGD gain")
    parser.add_argument(
        "--perturbation",
        type=float,
        default=0.02,
        help="SPGD perturbation amplitude",
    )
    parser.add_argument(
        "--auto-tune",
        action="store_true",
        help="Automatically refine gain and perturbation using simulation snapshots",
    )
    parser.add_argument(
        "--auto-tune-stages",
        type=int,
        default=2,
        help="Number of refinement stages when --auto-tune is enabled",
    )
    parser.add_argument(
        "--settling-time",
        type=float,
        default=0.0,
        help="Additional wait time after updating the drive before measuring",
    )
    parser.add_argument(
        "--noise-duration",
        type=float,
        default=0.0,
        help="Duration over which repeated measurements are averaged",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Load controller defaults from a JSON configuration file",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Optional path where the controller log will be written as JSON",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the intensity evolution using matplotlib",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=25,
        help="Print a status message every N iterations (0 disables logging)",
    )
    return parser.parse_args(argv)


def _select_backend(args: argparse.Namespace) -> AbstractBackend:
    if args.simulate or not args.host:
        print("Using simulated interferometer backend.")
        return SimulatedInterferometerBackend()
    return RedPitayaHardwareBackend(args.host, port=args.port)


def _build_parameters(args: argparse.Namespace, config: Dict[str, Number]) -> ControllerParameters:
    gain = float(config.get("gain", args.gain))
    perturbation = float(config.get("perturbation", args.perturbation))
    iterations = int(config.get("iterations", args.iterations))
    auto_tune = bool(config.get("auto_tune", args.auto_tune))
    auto_tune_stages = int(config.get("auto_tune_stages", args.auto_tune_stages))
    log_interval = int(config.get("log_interval", args.log_interval))
    settling_time = float(config.get("settling_time", args.settling_time))
    noise_duration = float(config.get("noise_duration", args.noise_duration))
    return ControllerParameters(
        gain=gain,
        perturbation=perturbation,
        iterations=iterations,
        auto_tune=auto_tune,
        auto_tune_stages=auto_tune_stages,
        log_interval=log_interval,
        settling_time=settling_time,
        noise_duration=noise_duration,
    )


def _save_results(path: Path, result: ControllerResult) -> None:
    payload = {
        "drive_history": [list(drive) for drive in result.drive_history],
        "intensity_history": list(result.intensity_history),
        "locking_efficiency": list(result.locking_efficiency),
        "transmission_sweeps": [
            {
                "channel": channel,
                "drive": list(drive_axis),
                "intensity": list(intensity_values),
            }
            for channel, drive_axis, intensity_values in result.transmission_sweeps
        ],
        "noise_spectrum": {
            "frequency": list(result.noise_spectrum[0]),
            "power": list(result.noise_spectrum[1]),
        },
    }
    with Path(path).expanduser().open("w", encoding="utf8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Stored results in {path}")


def _maybe_plot(result: ControllerResult) -> None:
    if not result.intensity_history:
        print("No data recorded; skipping plot.")
        return
    try:  # pragma: no cover - plotting is not exercised in tests
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"Unable to import matplotlib: {exc}")
        _print_ascii_plot(result)
        return
    xs = list(range(len(result.intensity_history)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Red Pitaya SPGD Lock Diagnostics")

    # Real-time detector intensity -------------------------------------------------
    ax = axes[0][0]
    ax.plot(xs, result.intensity_history, label="Intensity", color="tab:blue")
    ax.set_title("Detector intensity")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Intensity (a.u.)")
    ax.grid(True)

    # Locking efficiency ----------------------------------------------------------
    ax = axes[0][1]
    if result.locking_efficiency:
        ax.plot(xs, result.locking_efficiency, color="tab:green")
        ax.set_ylim(0.0, 1.05)
    ax.set_title("Locking efficiency")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Efficiency")
    ax.grid(True)

    # Transmission function -------------------------------------------------------
    ax = axes[1][0]
    if result.transmission_sweeps:
        for channel, drive_axis, intensity_values in result.transmission_sweeps:
            ax.plot(
                drive_axis,
                intensity_values,
                label=f"Channel {channel + 1}",
            )
    ax.set_title("Post-lock transmission function")
    ax.set_xlabel("Drive command (rad)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.grid(True)
    if result.transmission_sweeps:
        ax.legend()

    # Intensity noise -------------------------------------------------------------
    ax = axes[1][1]
    freqs, power = result.noise_spectrum
    if freqs and power:
        ax.semilogy(freqs, power, color="tab:red")
    ax.set_title("Intensity noise spectrum")
    ax.set_xlabel("Normalised frequency (1/iteration)")
    ax.set_ylabel("Power (a.u.)")
    ax.grid(True, which="both")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def _print_ascii_plot(result: ControllerResult, width: int = 80) -> None:
    samples = result.intensity_history
    if not samples:
        return
    if width <= 0:
        width = 80
    if len(samples) <= width:
        bucketed = list(samples)
    else:
        bucketed = []
        step = len(samples) / width
        for index in range(width):
            start = int(index * step)
            end = int((index + 1) * step)
            if end <= start:
                end = start + 1
            slice_ = samples[start:end]
            bucketed.append(sum(slice_) / len(slice_))
    minimum = min(bucketed)
    maximum = max(bucketed)
    levels = "▁▂▃▄▅▆▇█"
    if math.isclose(maximum, minimum):
        line = levels[0] * len(bucketed)
    else:
        scale = len(levels) - 1
        span = maximum - minimum
        line = "".join(
            levels[int(round(((value - minimum) / span) * scale))]
            for value in bucketed
        )
    print("ASCII intensity trend:")
    print(line)
    print(f"min={minimum:.4f}, max={maximum:.4f}")

    if result.locking_efficiency:
        last_eff = result.locking_efficiency[-1]
        print(f"Final locking efficiency: {last_eff:.3f}")

    if result.transmission_sweeps:
        for channel, drive_axis, intensity_values in result.transmission_sweeps:
            peak = max(intensity_values)
            trough = min(intensity_values)
            print(
                f"Channel {channel + 1} transmission span: "
                f"drive {drive_axis[0]:.4f}..{drive_axis[-1]:.4f}, "
                f"intensity {trough:.4f}..{peak:.4f}"
            )

    if result.noise_spectrum[0] and result.noise_spectrum[1]:
        peak_idx = max(range(len(result.noise_spectrum[1])), key=result.noise_spectrum[1].__getitem__)
        freq = result.noise_spectrum[0][peak_idx]
        power = result.noise_spectrum[1][peak_idx]
        print(f"Dominant noise component: freq={freq:.4f}, power={power:.4e}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> ControllerResult:
    args = _parse_args(argv)
    config = _load_json_config(args.config)
    params = _build_parameters(args, config)

    with _select_backend(args) as backend:
        controller = SPGDController(backend, params)
        result = controller.run()

    if args.save:
        _save_results(args.save, result)

    print(result.summary())

    if args.plot:
        _maybe_plot(result)

    return result


if __name__ == "__main__":
    main()
