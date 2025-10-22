#!/usr/bin/env python3
"""Example script that runs the NALM fiber laser simulation."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from nalmsim import NALMFiberLaserSimulation


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-trips", type=int, default=400,
                        help="Number of cavity round-trips to simulate (or the maximum"
                             " to allow when searching for mode-locking)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the random initial field")
    parser.add_argument("--pump-bias", type=float, default=0.0,
                        help="Additional bias applied to the gain medium")
    parser.add_argument("--store-every", type=int, default=10,
                        help="Store the intracavity field every N round-trips")
    parser.add_argument("--mode-lock", action="store_true",
                        help="Continue iterating until the intracavity energy converges")
    parser.add_argument("--lock-tolerance", type=float, default=5e-3,
                        help="Relative standard deviation threshold used to declare"
                             " mode-locking when --mode-lock is enabled")
    parser.add_argument("--lock-window", type=int, default=50,
                        help="Number of recent round-trips considered when assessing"
                             " mode-locking convergence")
    parser.add_argument("--min-round-trips", type=int, default=100,
                        help="Minimum number of round-trips to simulate before checking"
                             " for mode-locking convergence")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable plotting of the temporal and spectral evolution")
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
    if result.mode_locked is True:
        print("  Mode-locking       : converged")
    elif result.mode_locked is False:
        print("  Mode-locking       : not converged (max round-trips reached)")
    else:
        print("  Mode-locking       : not evaluated")


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

    ax_energy.plot(round_trips, energies)
    ax_energy.set_xlabel("Round-trip")
    ax_energy.set_ylabel("Energy (J)")
    ax_energy.set_title("Intracavity energy evolution")

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

    fig.tight_layout()
    plt.show()


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    sim = NALMFiberLaserSimulation(store_every=args.store_every)
    if args.mode_lock:
        result = sim.run_until_mode_locked(
            args.round_trips,
            seed=args.seed,
            pump_bias=args.pump_bias,
            min_round_trips=args.min_round_trips,
            energy_window=args.lock_window,
            relative_tolerance=args.lock_tolerance,
        )
    else:
        result = sim.run(args.round_trips, seed=args.seed, pump_bias=args.pump_bias)

    summarise(result)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save, result.field_history[-1])
        print(f"Saved final field to {args.save}")

    if not args.no_plot:
        plot_results(result)


if __name__ == "__main__":
    main()
