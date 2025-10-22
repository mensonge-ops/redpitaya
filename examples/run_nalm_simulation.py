#!/usr/bin/env python3
"""Example script that runs the NALM fiber laser simulation."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from nalmsim import NALMFiberLaserSimulation


# Default values used when parsing arguments so the example can be tuned in one place.
DEFAULT_CONFIGURATION = {
    "round_trips": 500,
    "seed": 1,
    "store_every": 10,
    "mode_lock": True,
    "no_plot": True,
}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-trips", type=int, default=DEFAULT_CONFIGURATION["round_trips"],
                        help="Number of cavity round-trips to simulate (or the maximum"
                             " to allow when searching for mode-locking)")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIGURATION["seed"],
                        help="Seed for the random initial field")
    parser.add_argument("--pump-bias", type=float, default=0.0,
                        help="Additional bias applied to the gain medium")
    parser.add_argument("--store-every", type=int, default=DEFAULT_CONFIGURATION["store_every"],
                        help="Store the intracavity field every N round-trips")
    parser.add_argument("--mode-lock", dest="mode_lock", action="store_true",
                        help="Continue iterating until the intracavity energy converges")
    parser.add_argument("--no-mode-lock", dest="mode_lock", action="store_false",
                        help="Disable the convergence search and run a fixed number of round-trips")
    parser.add_argument("--lock-tolerance", type=float, default=5e-3,
                        help="Relative standard deviation threshold used to declare"
                             " mode-locking when --mode-lock is enabled")
    parser.add_argument("--lock-window", type=int, default=50,
                        help="Number of recent round-trips considered when assessing"
                             " mode-locking convergence")
    parser.add_argument("--contrast-threshold", type=float, default=15.0,
                        help="Minimum peak-to-average power ratio required for"
                             " the mode-lock detector")
    parser.add_argument("--tbp-threshold", type=float, default=0.65,
                        help="Maximum RMS time-bandwidth product tolerated when"
                             " declaring mode-locking")
    parser.add_argument("--peak-power-threshold", type=float, default=80.0,
                        help="Minimum intracavity peak power (W) considered a"
                             " clean pulse")
    parser.add_argument("--min-round-trips", type=int, default=100,
                        help="Minimum number of round-trips to simulate before checking"
                             " for mode-locking convergence")
    parser.add_argument("--plot", dest="no_plot", action="store_false",
                        help="Enable plotting of the temporal and spectral evolution")
    parser.add_argument("--no-plot", dest="no_plot", action="store_true",
                        help="Disable plotting of the temporal and spectral evolution")
    parser.add_argument("--adaptive-pump", dest="adaptive_pump", action="store_true",
                        help="Enable adaptive pump control to help reach mode-locking")
    parser.add_argument("--no-adaptive-pump", dest="adaptive_pump", action="store_false",
                        help="Disable adaptive pump control")
    parser.set_defaults(
        adaptive_pump=None,
        mode_lock=DEFAULT_CONFIGURATION["mode_lock"],
        no_plot=DEFAULT_CONFIGURATION["no_plot"],
    )
    parser.add_argument("--target-energy", type=float, default=None,
                        help="Desired intracavity energy level when adaptive pumping is enabled")
    parser.add_argument("--pump-step", type=float, default=0.1,
                        help="Adjustment applied to the pump bias controller each round-trip")
    parser.add_argument("--pump-min", type=float, default=-0.5,
                        help="Lower bound applied to the pump bias when adapting")
    parser.add_argument("--pump-max", type=float, default=2.0,
                        help="Upper bound applied to the pump bias when adapting")
    parser.add_argument("--pump-smoothing", type=float, default=0.75,
                        help="Smoothing factor for the pump controller error signal (0-1)")
    parser.add_argument("--save", type=Path, default=None,
                        help="Optional path where the final field will be stored as a NumPy file")
    return parser.parse_args(argv)


def summarise(result):
    last = result.history[-1]
    print("Final round-trip:")
    print(f"  number             : {last.round_trip}")
    print(f"  intracavity energy : {last.intracavity_energy:.3e} J")
    print(f"  output energy      : {last.output_energy:.3e} J")
    print(f"  NALM transmission  : {last.nalm_transmission:.3f}")
    print(f"  CW/CCW energies    : {last.cw_energy:.3e} / {last.ccw_energy:.3e} J")
    print(f"  Gain exponent      : {last.gain:.3f}")
    print(f"  Pump bias          : {last.pump_bias:.3f}")
    print(f"  Peak power         : {last.peak_power:.3e} W")
    print(f"  Pulse duration     : {last.pulse_duration * 1e12:.3f} ps (RMS)")
    print(f"  Spectral width     : {last.spectral_width * 1e-12:.3f} THz (RMS)")
    print(f"  Time-bandwidth prod: {last.time_bandwidth_product:.3f}")
    print(f"  Pulse contrast     : {last.pulse_contrast:.1f}")
    if result.mode_locked is True:
        print("  Mode-locking       : converged")
    elif result.mode_locked is False:
        print("  Mode-locking       : not converged (max round-trips reached)")
    else:
        print("  Mode-locking       : not evaluated")

    if result.mode_lock_report is not None:
        report = result.mode_lock_report
        status = {
            True: "OK",
            False: "not met",
        }
        print("\nMode-lock assessment window:")
        print(f"  Energy stability   : {report.energy_std:.3e} (limit {report.energy_tolerance:.3e})"
              f" -> {status[report.met_energy_stability]}")
        print(f"  Peak contrast      : {report.representative_contrast:.2f}"
              f" (limit {report.contrast_threshold:.2f}) -> {status[report.met_contrast]}"
              f" | mean {report.mean_contrast:.2f}")
        print(f"  Time-bandwidth     : {report.representative_time_bandwidth_product:.3f}"
              f" (limit {report.tbp_threshold:.3f}) -> {status[report.met_tbp]}"
              f" | mean {report.mean_time_bandwidth_product:.3f}")
        print(f"  Peak power         : {report.representative_peak_power:.3e} W"
              f" (limit {report.peak_power_threshold:.3e} W) -> {status[report.met_peak_power]}"
              f" | mean {report.mean_peak_power:.3e} W")


def plot_results(result):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("matplotlib is required for plotting") from exc

    last_field = result.field_history[-1]
    time_ps = result.time_axis * 1e12
    spectrum = result.frequency_axis / (2 * np.pi)
    spectral_density = np.abs(np.fft.fftshift(np.fft.fft(last_field))) ** 2

    history = result.history
    round_trips = [entry.round_trip for entry in history]
    energies = [entry.intracavity_energy for entry in history]
    transmissions = [entry.nalm_transmission for entry in history]
    pump_biases = [entry.pump_bias for entry in history]

    intensity_map = np.abs(result.field_history) ** 2
    spectrum_map = np.abs(
        np.fft.fftshift(
            np.fft.fft(result.output_history, axis=1),
            axes=1,
        )
    ) ** 2

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 2)

    ax_pulse = fig.add_subplot(gs[0, 0])
    ax_spec = fig.add_subplot(gs[0, 1])
    ax_energy = fig.add_subplot(gs[1, 0])
    ax_trans = fig.add_subplot(gs[1, 1])
    ax_pulse_evo = fig.add_subplot(gs[2, 0])
    ax_spec_evo = fig.add_subplot(gs[2, 1])

    ax_pulse.plot(time_ps, np.abs(last_field) ** 2)
    ax_pulse.set_xlabel("Time (ps)")
    ax_pulse.set_ylabel("Power (W)")
    ax_pulse.set_title("Final intracavity pulse")

    ax_spec.plot(np.fft.fftshift(spectrum) * 1e-12, spectral_density)
    ax_spec.set_xlabel("Frequency offset (THz)")
    ax_spec.set_ylabel("Spectral density (a.u.)")
    ax_spec.set_title("Final output spectrum")

    ax_energy.plot(round_trips, energies, label="Energy")
    ax_energy.set_xlabel("Round-trip")
    ax_energy.set_ylabel("Energy (J)")
    ax_energy.set_title("Intracavity energy evolution")

    if any(pump_biases):
        ax_pump = ax_energy.twinx()
        ax_pump.plot(round_trips, pump_biases, color="tab:red", linestyle="--", label="Pump bias")
        ax_pump.set_ylabel("Pump bias (a.u.)")
        lines, labels = ax_energy.get_legend_handles_labels()
        lines2, labels2 = ax_pump.get_legend_handles_labels()
        ax_energy.legend(lines + lines2, labels + labels2, loc="best")

    ax_trans.plot(round_trips, transmissions)
    ax_trans.set_xlabel("Round-trip")
    ax_trans.set_ylabel("Transmission")
    ax_trans.set_title("NALM transmission evolution")

    round_min = int(result.stored_round_trips[0])
    round_max = int(result.stored_round_trips[-1])
    if round_max == round_min:
        round_max += 1

    extent = [time_ps[0], time_ps[-1], round_min, round_max]
    im = ax_pulse_evo.imshow(
        intensity_map,
        aspect="auto",
        extent=extent,
        origin="lower",
        interpolation="nearest",
    )
    ax_pulse_evo.set_xlabel("Time (ps)")
    ax_pulse_evo.set_ylabel("Stored round-trip")
    ax_pulse_evo.set_title("Pulse evolution")
    fig.colorbar(im, ax=ax_pulse_evo, label="Power (W)")

    spectral_axis = np.fft.fftshift(spectrum) * 1e-12
    spectrum_extent = [spectral_axis[0], spectral_axis[-1], round_min, round_max]
    im_spec = ax_spec_evo.imshow(
        spectrum_map,
        aspect="auto",
        extent=spectrum_extent,
        origin="lower",
        interpolation="nearest",
    )
    ax_spec_evo.set_xlabel("Frequency offset (THz)")
    ax_spec_evo.set_ylabel("Stored round-trip")
    ax_spec_evo.set_title("Spectrum evolution")
    fig.colorbar(im_spec, ax=ax_spec_evo, label="Spectral density (a.u.)")

    if result.mode_lock_report is not None:
        report = result.mode_lock_report
        status = "Mode-locked" if result.mode_locked else "Not mode-locked"
        fig.suptitle(
            f"{status}: σ_E/⟨E⟩={report.energy_std:.2e}, "
            f"contrast={report.representative_contrast:.1f}, "
            f"TBP={report.representative_time_bandwidth_product:.3f}",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
    else:
        fig.tight_layout()

    if result.mode_locked and result.field_history.shape[0] >= 2:
        num_traces = min(6, result.field_history.shape[0])
        window = intensity_map[-num_traces:]
        separation = float(np.max(window))
        if not np.isfinite(separation) or separation <= 0.0:
            separation = 1.0
        separation *= 1.15

        fig_seq, ax_seq = plt.subplots(figsize=(10, 4))
        for offset_index, idx in enumerate(range(-num_traces, 0)):
            trace = intensity_map[idx]
            label = f"RT {int(result.stored_round_trips[idx])}"
            ax_seq.plot(time_ps, trace + separation * offset_index, label=label)

        ax_seq.set_xlabel("Time (ps)")
        ax_seq.set_ylabel("Relative power (offset)")
        ax_seq.set_title("Stable pulse sequence after mode-locking")
        ax_seq.legend(loc="upper right", frameon=False)
        ax_seq.set_ylim(bottom=-0.05 * separation)
        if result.mode_lock_report is not None:
            report = result.mode_lock_report
            fig_seq.suptitle(
                f"Stable pulse sequence (contrast {report.representative_contrast:.1f},"
                f" TBP {report.representative_time_bandwidth_product:.3f})",
                fontsize=11,
            )
            fig_seq.tight_layout(rect=[0, 0, 1, 0.9])
        else:
            fig_seq.tight_layout()

    plt.show()


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    sim = NALMFiberLaserSimulation(store_every=args.store_every)
    adaptive_pump = args.adaptive_pump
    if adaptive_pump is None:
        adaptive_pump = args.mode_lock

    pump_kwargs = dict(
        adaptive_pump=adaptive_pump,
        target_energy=args.target_energy,
        pump_adjustment=args.pump_step,
        pump_min=args.pump_min,
        pump_max=args.pump_max,
        pump_smoothing=args.pump_smoothing,
    )

    if args.mode_lock:
        result = sim.run_until_mode_locked(
            args.round_trips,
            seed=args.seed,
            pump_bias=args.pump_bias,
            min_round_trips=args.min_round_trips,
            energy_window=args.lock_window,
            relative_tolerance=args.lock_tolerance,
            contrast_threshold=args.contrast_threshold,
            tbp_threshold=args.tbp_threshold,
            peak_power_threshold=args.peak_power_threshold,
            **pump_kwargs,
        )
    else:
        result = sim.run(
            args.round_trips,
            seed=args.seed,
            pump_bias=args.pump_bias,
            **pump_kwargs,
        )

    summarise(result)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save, result.field_history[-1])
        print(f"Saved final field to {args.save}")

    if not args.no_plot:
        plot_results(result)


if __name__ == "__main__":
    main()
