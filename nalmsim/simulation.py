"""High-level orchestration for the multi-stage NALM mode-locking workflow."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Sequence

import numpy as np

from .components import (
    BandpassFilter,
    FiberSegment,
    GainFiber,
    NALM,
    OutputCoupler,
    Pulse,
    SaturableAbsorber,
    TemporalGrid,
)

__all__ = [
    "ModeLockedCavity",
    "ModeLockingCriteria",
    "ModeLockingReport",
    "SimulationHistoryEntry",
    "SimulationResult",
    "SimulationStagePlan",
    "ModeLockingWorkflow",
    "create_initial_pulse",
    "build_reference_stage",
    "build_yb_mapping_stage",
    "build_target_nalm_stage",
]


# ---------------------------------------------------------------------------
# Diagnostic and reporting helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ModeLockingCriteria:
    """Numerical thresholds used to declare a mode-locked state."""

    contrast_min: float = 6.0
    tbp_range: tuple[float, float] = (0.25, 1.5)
    peak_power_min: float = 80.0
    energy_stability: float = 5e-3
    window: int = 200


@dataclass(slots=True)
class ModeLockingReport:
    """Summary of the mode-locking state at the end of a simulation stage."""

    locked: bool
    round_trip: int
    energy_mean: float
    energy_std: float
    representative_peak_power: float
    representative_contrast: float
    representative_tbp: float
    output_energy: float
    criteria: ModeLockingCriteria
    notes: str = ""


@dataclass(slots=True)
class SimulationHistoryEntry:
    round_trip: int
    intracavity_energy: float
    output_energy: float
    pulse: Pulse
    output_pulse: Pulse


@dataclass(slots=True)
class SimulationResult:
    stage_name: str
    history: List[SimulationHistoryEntry]
    diagnostics: Dict[str, List[float]]
    report: ModeLockingReport
    final_pulse: Pulse
    final_output: Pulse


# ---------------------------------------------------------------------------
# Core propagation logic
# ---------------------------------------------------------------------------


def create_initial_pulse(grid: TemporalGrid, seed: int, amplitude: float = 1e-6) -> Pulse:
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=grid.points) + 1j * rng.normal(size=grid.points)
    return Pulse(grid, amplitude * noise.astype(np.complex128))


@dataclass(slots=True)
class ModeLockedCavity:
    """Cavity composed of a sequence of components."""

    grid: TemporalGrid
    components: Sequence[object]
    warmup: int = 500

    def propagate_once(self, pulse: Pulse) -> tuple[Pulse, Pulse]:
        field = pulse
        extracted = None
        for component in self.components:
            if isinstance(component, OutputCoupler):
                field, extracted = component.propagate(field)
            else:
                field = component.propagate(field)
        if extracted is None:
            extracted = Pulse(self.grid, np.zeros_like(field.field))
        return field, extracted

    def run(
        self,
        initial_pulse: Pulse,
        *,
        round_trips: int,
        store_every: int,
        criteria: ModeLockingCriteria,
        mode_lock: bool,
    ) -> SimulationResult:
        pulse = initial_pulse.copy()
        diagnostics: Dict[str, List[float]] = {
            "intracavity_energy": [],
            "output_energy": [],
            "peak_power": [],
            "contrast": [],
            "tbp": [],
        }
        history: List[SimulationHistoryEntry] = []
        stored_rounds: Deque[SimulationHistoryEntry] = deque(maxlen=max(10, criteria.window // 2))
        monitor = _ModeLockingMonitor(criteria)

        final_round = round_trips
        for round_trip in range(1, round_trips + 1):
            pulse, output = self.propagate_once(pulse)
            energy = pulse.energy()
            output_energy = output.energy()
            peak = pulse.peak_power()
            contrast = _contrast(pulse)
            tbp = pulse.time_bandwidth_product()

            diagnostics["intracavity_energy"].append(energy)
            diagnostics["output_energy"].append(output_energy)
            diagnostics["peak_power"].append(peak)
            diagnostics["contrast"].append(contrast)
            diagnostics["tbp"].append(tbp)

            entry = SimulationHistoryEntry(
                round_trip=round_trip,
                intracavity_energy=energy,
                output_energy=output_energy,
                pulse=pulse.copy(),
                output_pulse=output.copy(),
            )
            stored_rounds.append(entry)
            if store_every > 0 and round_trip % store_every == 0:
                history.append(entry)

            locked, report = monitor.update(round_trip, pulse, output_energy)
            if locked and mode_lock and round_trip >= max(self.warmup, criteria.window):
                final_round = round_trip
                break

        if not history:
            history.extend(list(stored_rounds)[-3:])
        elif history[-1].round_trip != final_round:
            history.append(stored_rounds[-1])

        final_entry = stored_rounds[-1]
        final_pulse = final_entry.pulse.copy()
        final_output = final_entry.output_pulse.copy()
        report = monitor.final_report(final_round)
        return SimulationResult(
            stage_name="",
            history=history,
            diagnostics=diagnostics,
            report=report,
            final_pulse=final_pulse,
            final_output=final_output,
        )


class _ModeLockingMonitor:
    def __init__(self, criteria: ModeLockingCriteria) -> None:
        self.criteria = criteria
        self.energies: Deque[float] = deque(maxlen=criteria.window)
        self.peak_powers: Deque[float] = deque(maxlen=criteria.window)
        self.contrasts: Deque[float] = deque(maxlen=criteria.window)
        self.tbps: Deque[float] = deque(maxlen=criteria.window)
        self.output_energies: Deque[float] = deque(maxlen=criteria.window)
        self._locked = False
        self._last_report: Optional[ModeLockingReport] = None

    def update(self, round_trip: int, pulse: Pulse, output_energy: float) -> tuple[bool, ModeLockingReport]:
        energy = pulse.energy()
        peak = pulse.peak_power()
        contrast = _contrast(pulse)
        tbp = pulse.time_bandwidth_product()

        self.energies.append(energy)
        self.peak_powers.append(peak)
        self.contrasts.append(contrast)
        self.tbps.append(tbp)
        self.output_energies.append(output_energy)

        report = self._build_report(round_trip)
        self._last_report = report
        self._locked = report.locked
        return self._locked, report

    def _build_report(self, round_trip: int) -> ModeLockingReport:
        if len(self.energies) == 0:
            return ModeLockingReport(
                locked=False,
                round_trip=round_trip,
                energy_mean=0.0,
                energy_std=0.0,
                representative_peak_power=0.0,
                representative_contrast=0.0,
                representative_tbp=0.0,
                output_energy=0.0,
                criteria=self.criteria,
                notes="insufficient data",
            )

        energy_mean = float(np.mean(self.energies))
        energy_std = float(np.std(self.energies))
        peak = float(np.median(self.peak_powers))
        contrast = float(np.median(self.contrasts))
        tbp = float(np.median(self.tbps))
        output_energy = float(np.median(self.output_energies))

        energy_ok = energy_mean > 0.0 and energy_std / energy_mean < self.criteria.energy_stability
        contrast_ok = contrast >= self.criteria.contrast_min
        tbp_ok = self.criteria.tbp_range[0] <= tbp <= self.criteria.tbp_range[1]
        peak_ok = peak >= self.criteria.peak_power_min
        locked = energy_ok and contrast_ok and tbp_ok and peak_ok

        notes: List[str] = []
        if not energy_ok:
            notes.append("energy unstable")
        if not contrast_ok:
            notes.append("contrast below threshold")
        if not tbp_ok:
            notes.append("TBP outside range")
        if not peak_ok:
            notes.append("peak power too low")

        return ModeLockingReport(
            locked=locked,
            round_trip=round_trip,
            energy_mean=energy_mean,
            energy_std=energy_std,
            representative_peak_power=peak,
            representative_contrast=contrast,
            representative_tbp=tbp,
            output_energy=output_energy,
            criteria=self.criteria,
            notes=", ".join(notes) if notes else "locked",
        )

    def final_report(self, final_round: int) -> ModeLockingReport:
        if self._last_report is None:
            return self._build_report(final_round)
        return ModeLockingReport(
            locked=self._last_report.locked,
            round_trip=final_round,
            energy_mean=self._last_report.energy_mean,
            energy_std=self._last_report.energy_std,
            representative_peak_power=self._last_report.representative_peak_power,
            representative_contrast=self._last_report.representative_contrast,
            representative_tbp=self._last_report.representative_tbp,
            output_energy=self._last_report.output_energy,
            criteria=self._last_report.criteria,
            notes=self._last_report.notes,
        )


def _contrast(pulse: Pulse) -> float:
    avg_power = pulse.energy() / pulse.grid.window
    if avg_power <= 0.0:
        return 0.0
    return float(pulse.peak_power() / avg_power)


# ---------------------------------------------------------------------------
# Workflow orchestration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SimulationStagePlan:
    name: str
    builder: Callable[[TemporalGrid], ModeLockedCavity]
    round_trips: int
    store_every: int
    criteria: ModeLockingCriteria
    mode_lock: bool = True


@dataclass(slots=True)
class ModeLockingWorkflow:
    grid: TemporalGrid
    stages: Sequence[SimulationStagePlan]

    def run(self, seed: int) -> List[SimulationResult]:
        results: List[SimulationResult] = []
        pulse = create_initial_pulse(self.grid, seed)
        for plan in self.stages:
            cavity = plan.builder(self.grid)
            result = cavity.run(
                pulse,
                round_trips=plan.round_trips,
                store_every=plan.store_every,
                criteria=plan.criteria,
                mode_lock=plan.mode_lock,
            )
            result.stage_name = plan.name
            results.append(result)
            pulse = result.final_pulse.copy()
        return results


# ---------------------------------------------------------------------------
# Stage builders following the requested workflow
# ---------------------------------------------------------------------------


def build_reference_stage(grid: TemporalGrid) -> ModeLockedCavity:
    """Stage 1: reproduce a textbook NALM passively mode-locked cavity."""

    main_gain = GainFiber(
        length=1.0,
        beta2=2.3e-26,
        gamma=2.1e-3,
        loss=4.5e-5,
        steps=30,
        small_signal_gain=4.2,
        saturation_energy=45e-9,
        bandwidth_fwhm=25e12,
        gain_clamp=(0.0, 5.0),
    )
    passive_fiber = FiberSegment(
        length=2.5,
        beta2=2.3e-26,
        gamma=2.5e-3,
        loss=4.0e-5,
        steps=40,
    )
    filter_stage = BandpassFilter(bandwidth_fwhm=18e12)
    absorber = SaturableAbsorber(modulation_depth=0.18, saturation_power=4e3, nonsaturable_loss=0.04)
    output = OutputCoupler(ratio=0.1)

    cw_gain = GainFiber(
        length=0.7,
        beta2=2.3e-26,
        gamma=2.7e-3,
        loss=4.0e-5,
        steps=24,
        small_signal_gain=2.1,
        saturation_energy=25e-9,
        bandwidth_fwhm=28e12,
        gain_clamp=(0.0, 3.0),
    )
    ccw_gain = GainFiber(
        length=0.7,
        beta2=2.3e-26,
        gamma=2.7e-3,
        loss=4.0e-5,
        steps=24,
        small_signal_gain=2.1,
        saturation_energy=25e-9,
        bandwidth_fwhm=28e12,
        gain_clamp=(0.0, 3.0),
    )
    cw_loop = [cw_gain, FiberSegment(length=4.0, beta2=2.3e-26, gamma=3.0e-3, loss=5e-5, steps=40)]
    ccw_loop = [FiberSegment(length=4.0, beta2=2.3e-26, gamma=3.0e-3, loss=5e-5, steps=40), ccw_gain]
    nalm = NALM(coupling_ratio=0.55, cw_path=cw_loop, ccw_path=ccw_loop)

    components: List[object] = [main_gain, filter_stage, absorber, passive_fiber, output, nalm]
    return ModeLockedCavity(grid=grid, components=components, warmup=600)


def build_yb_mapping_stage(grid: TemporalGrid) -> ModeLockedCavity:
    """Stage 2: map Yb amplifier characteristics onto the reference cavity."""

    gain = GainFiber(
        length=1.2,
        beta2=2.1e-26,
        gamma=2.0e-3,
        loss=4.0e-5,
        steps=36,
        small_signal_gain=4.8,
        saturation_energy=65e-9,
        bandwidth_fwhm=32e12,
        gain_clamp=(0.0, 5.5),
    )
    passive = FiberSegment(
        length=3.0,
        beta2=2.1e-26,
        gamma=2.4e-3,
        loss=4.5e-5,
        steps=48,
    )
    filter_stage = BandpassFilter(bandwidth_fwhm=16e12)
    absorber = SaturableAbsorber(modulation_depth=0.2, saturation_power=3.2e3, nonsaturable_loss=0.05)
    output = OutputCoupler(ratio=0.12)

    cw_loop = [
        GainFiber(
            length=0.8,
            beta2=2.1e-26,
            gamma=2.8e-3,
            loss=4.5e-5,
            steps=28,
            small_signal_gain=2.4,
            saturation_energy=30e-9,
            bandwidth_fwhm=30e12,
            gain_clamp=(0.0, 3.5),
        ),
        FiberSegment(length=4.5, beta2=2.1e-26, gamma=3.2e-3, loss=5e-5, steps=48),
    ]
    ccw_loop = [
        FiberSegment(length=4.5, beta2=2.1e-26, gamma=3.2e-3, loss=5e-5, steps=48),
        GainFiber(
            length=0.8,
            beta2=2.1e-26,
            gamma=2.8e-3,
            loss=4.5e-5,
            steps=28,
            small_signal_gain=2.4,
            saturation_energy=30e-9,
            bandwidth_fwhm=30e12,
            gain_clamp=(0.0, 3.5),
        ),
    ]
    nalm = NALM(coupling_ratio=0.57, cw_path=cw_loop, ccw_path=ccw_loop)

    components: List[object] = [gain, filter_stage, absorber, passive, output, nalm]
    return ModeLockedCavity(grid=grid, components=components, warmup=800)


def build_target_nalm_stage(grid: TemporalGrid) -> ModeLockedCavity:
    """Stage 3: final cavity using the project-specific parameters."""

    main_gain = GainFiber(
        length=1.4,
        beta2=2.05e-26,
        gamma=2.1e-3,
        loss=4.5e-5,
        steps=48,
        small_signal_gain=5.2,
        saturation_energy=70e-9,
        bandwidth_fwhm=34e12,
        gain_clamp=(0.0, 6.0),
    )
    passive = FiberSegment(
        length=3.5,
        beta2=2.05e-26,
        gamma=2.5e-3,
        loss=4.8e-5,
        steps=60,
    )
    dispersion_comp = FiberSegment(
        length=0.8,
        beta2=-1.5e-26,
        gamma=0.8e-3,
        loss=3.0e-5,
        steps=24,
    )
    filter_stage = BandpassFilter(bandwidth_fwhm=14e12)
    absorber = SaturableAbsorber(modulation_depth=0.22, saturation_power=2.6e3, nonsaturable_loss=0.05)
    output = OutputCoupler(ratio=0.15)

    cw_loop = [
        GainFiber(
            length=0.9,
            beta2=2.05e-26,
            gamma=3.0e-3,
            loss=4.5e-5,
            steps=30,
            small_signal_gain=2.8,
            saturation_energy=28e-9,
            bandwidth_fwhm=32e12,
            gain_clamp=(0.0, 3.8),
        ),
        FiberSegment(length=5.0, beta2=2.05e-26, gamma=3.4e-3, loss=5e-5, steps=60),
    ]
    ccw_loop = [
        FiberSegment(length=5.0, beta2=2.05e-26, gamma=3.4e-3, loss=5e-5, steps=60),
        GainFiber(
            length=0.9,
            beta2=2.05e-26,
            gamma=3.0e-3,
            loss=4.5e-5,
            steps=30,
            small_signal_gain=2.8,
            saturation_energy=28e-9,
            bandwidth_fwhm=32e12,
            gain_clamp=(0.0, 3.8),
        ),
    ]
    nalm = NALM(coupling_ratio=0.6, cw_path=cw_loop, ccw_path=ccw_loop)

    components: List[object] = [main_gain, filter_stage, passive, absorber, output, dispersion_comp, nalm]
    return ModeLockedCavity(grid=grid, components=components, warmup=1000)
