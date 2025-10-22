"""Run the three-stage NALM mode-locking workflow described in the README.

The script performs the following steps:

1. Reproduce a textbook NALM cavity to validate the split-step propagation.
2. Map the parameters of an ytterbium-doped amplifier to the cavity model.
3. Assemble the target laser structure and iterate until it mode-locks.

Each stage reports key diagnostics and, unless ``--no-plot`` is given, renders
both the temporal pulse shape and the optical spectrum.  The final stage also
plots a short sequence of consecutive pulses so that a stable mode-locked train
can be visually confirmed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np

from nalmsim import (
    ModeLockingCriteria,
    ModeLockingWorkflow,
    SimulationResult,
    SimulationStagePlan,
    TemporalGrid,
    build_reference_stage,
    build_target_nalm_stage,
    build_yb_mapping_stage,
)

DEFAULTS = {
    "round_trips_final": 5000,
    "round_trips_reference": 2000,
    "round_trips_mapping": 3000,
    "store_every_final": 10000,
    "store_every_reference": 200,
    "store_every_mapping": 200,
    "seed": 1,
    "mode_lock": True,
    "no_plot": False,
    "grid_points": 4096,
    "time_window": 50e-12,
    "contrast_min": 6.0,
    "tbp_min": 0.25,
    "tbp_max": 1.5,
    "peak_power_min": 80.0,
    "energy_stability": 5e-3,
    "analysis_window": 400,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-trips", type=int, default=DEFAULTS["round_trips_final"],
                        help="Round-trips for the final stage (target cavity)")
    parser.add_argument("--round-trips-reference", type=int,
                        default=DEFAULTS["round_trips_reference"],
                        help="Round-trips for the reference reproduction stage")
    parser.add_argument("--round-trips-mapping", type=int,
                        default=DEFAULTS["round_trips_mapping"],
                        help="Round-trips for the ytterbium mapping stage")
    parser.add_argument("--store-every", type=int, default=DEFAULTS["store_every_final"],
                        help="Store the final-stage field every N round-trips")
    parser.add_argument("--store-every-reference", type=int,
                        default=DEFAULTS["store_every_reference"],
                        help="Storage cadence for the reference stage")
    parser.add_argument("--store-every-mapping", type=int,
                        default=DEFAULTS["store_every_mapping"],
                        help="Storage cadence for the mapping stage")
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"],
                        help="Random seed used for the initial noise field")
    parser.add_argument("--grid-points", type=int, default=DEFAULTS["grid_points"],
                        help="Number of temporal samples in the simulation grid")
    parser.add_argument("--time-window", type=float, default=DEFAULTS["time_window"],
                        help="Total simulation window in seconds")
    parser.add_argument("--mode-lock", dest="mode_lock", action="store_true",
                        help="Allow early termination when the criteria are met")
    parser.add_argument("--no-mode-lock", dest="mode_lock", action="store_false",
                        help="Always run the full number of round-trips")
    parser.add_argument("--contrast-min", type=float, default=DEFAULTS["contrast_min"],
                        help="Minimum peak-to-average contrast required for mode-locking")
    parser.add_argument("--tbp-min", type=float, default=DEFAULTS["tbp_min"],
                        help="Lower bound on the time-bandwidth product")
    parser.add_argument("--tbp-max", type=float, default=DEFAULTS["tbp_max"],
                        help="Upper bound on the time-bandwidth product")
    parser.add_argument("--peak-power-min", type=float, default=DEFAULTS["peak_power_min"],
                        help="Minimum intracavity peak power (W)")
    parser.add_argument("--energy-stability", type=float, default=DEFAULTS["energy_stability"],
                        help="Maximum relative standard deviation of the energy window")
    parser.add_argument("--analysis-window", type=int, default=DEFAULTS["analysis_window"],
                        help="Number of recent round-trips considered by the monitor")
    parser.add_argument("--no-plot", action="store_true", default=DEFAULTS["no_plot"],
                        help="Disable plotting of temporal and spectral results")
    parser.add_argument("--save-prefix", type=Path, default=None,
                        help="Optional prefix for saving diagnostic data as NPZ files")
    parser.set_defaults(mode_lock=DEFAULTS["mode_lock"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    grid = TemporalGrid(points=args.grid_points, window=args.time_window)
    criteria_final = ModeLockingCriteria(
        contrast_min=args.contrast_min,
        tbp_range=(args.tbp_min, args.tbp_max),
        peak_power_min=args.peak_power_min,
        energy_stability=args.energy_stability,
        window=args.analysis_window,
    )
    criteria_reference = ModeLockingCriteria(
        contrast_min=max(4.5, 0.8 * args.contrast_min),
        tbp_range=(args.tbp_min * 0.8, args.tbp_max * 1.2),
        peak_power_min=0.6 * args.peak_power_min,
        energy_stability=args.energy_stability * 1.5,
        window=min(args.analysis_window, 250),
    )
    criteria_mapping = ModeLockingCriteria(
        contrast_min=max(5.0, 0.9 * args.contrast_min),
        tbp_range=(args.tbp_min * 0.9, args.tbp_max * 1.1),
        peak_power_min=0.8 * args.peak_power_min,
        energy_stability=args.energy_stability,
        window=min(args.analysis_window, 300),
    )

    plans = [
        SimulationStagePlan(
            name="reference",
            builder=build_reference_stage,
            round_trips=args.round_trips_reference,
            store_every=max(1, args.store_every_reference),
            criteria=criteria_reference,
            mode_lock=args.mode_lock,
        ),
        SimulationStagePlan(
            name="yb-mapping",
            builder=build_yb_mapping_stage,
            round_trips=args.round_trips_mapping,
            store_every=max(1, args.store_every_mapping),
            criteria=criteria_mapping,
            mode_lock=args.mode_lock,
        ),
        SimulationStagePlan(
            name="target",
            builder=build_target_nalm_stage,
            round_trips=args.round_trips,
            store_every=max(1, args.store_every),
            criteria=criteria_final,
            mode_lock=args.mode_lock,
        ),
    ]

    workflow = ModeLockingWorkflow(grid=grid, stages=plans)
    results = workflow.run(seed=args.seed)

    for result in results:
        report_stage(result)

    if args.save_prefix is not None:
        save_results(args.save_prefix, results)

    if not args.no_plot:
        try:
            plot_results(results)
        except ImportError as exc:  # pragma: no cover - optional dependency
            print(f"Plotting skipped because matplotlib is unavailable: {exc}")


def report_stage(result: SimulationResult) -> None:
    report = result.report
    stage = result.stage_name
    status = "locked" if report.locked else f"not locked ({report.notes})"
    print(f"[{stage}] round-trip {report.round_trip}")
    print(f"  status           : {status}")
    print(f"  mean energy      : {report.energy_mean:.3e} J")
    print(f"  energy deviation : {report.energy_std:.3e} J")
    print(f"  peak power       : {report.representative_peak_power:.3e} W")
    print(f"  contrast         : {report.representative_contrast:.2f}")
    print(f"  time-bandwidth   : {report.representative_tbp:.3f}")
    print(f"  output energy    : {report.output_energy:.3e} J")


def save_results(prefix: Path, results: Iterable[SimulationResult]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for result in results:
        stage = result.stage_name
        filename = prefix.with_name(f"{prefix.name}_{stage}.npz")
        history_rounds = np.array([entry.round_trip for entry in result.history], dtype=float)
        history_energy = np.array([entry.intracavity_energy for entry in result.history], dtype=float)
        field = result.final_pulse.field
        output = result.final_output.field
        np.savez(
            filename,
            round_trip=history_rounds,
            intracavity_energy=history_energy,
            final_pulse=field,
            final_output=output,
            grid_time=result.final_pulse.grid.time,
        )


def plot_results(results: Iterable[SimulationResult]) -> None:
    import matplotlib.pyplot as plt

    results = list(results)
    for result in results:
        pulse = result.final_pulse
        grid = pulse.grid
        intensity = np.abs(pulse.field) ** 2
        freq = np.fft.fftshift(np.fft.fftfreq(grid.points, d=grid.dt))
        spectrum = pulse.spectral_intensity()
        spectrum = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(grid.time * 1e12, intensity)
        axes[0].set_title(f"{result.stage_name}: temporal intensity")
        axes[0].set_xlabel("Time (ps)")
        axes[0].set_ylabel("Power (W)")
        axes[1].plot(freq / 1e12, spectrum)
        axes[1].set_title(f"{result.stage_name}: optical spectrum")
        axes[1].set_xlabel("Frequency offset (THz)")
        axes[1].set_ylabel("Relative power (a.u.)")
        fig.tight_layout()

    final = results[-1]
    if final.history:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        pulses: List[np.ndarray] = [
            np.abs(entry.pulse.field) ** 2 for entry in final.history[-min(6, len(final.history)) :]
        ]
        matrix = np.stack(pulses)
        extent = [
            final.final_pulse.grid.time[0] * 1e12,
            final.final_pulse.grid.time[-1] * 1e12,
            final.history[-len(pulses)].round_trip,
            final.history[-1].round_trip,
        ]
        im = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="magma",
        )
        ax.set_title("Target stage: consecutive pulses")
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Round-trip index")
        fig.colorbar(im, ax=ax, label="Power (W)")
        fig.tight_layout()

    plt.show()


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
