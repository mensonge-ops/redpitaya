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
                        help="Number of cavity round-trips to simulate")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the random initial field")
    parser.add_argument("--pump-bias", type=float, default=0.0,
                        help="Additional bias applied to the gain medium")
    parser.add_argument("--store-every", type=int, default=10,
                        help="Store the intracavity field every N round-trips")
    parser.add_argument("--plot", action="store_true",
                        help="Plot the temporal and spectral evolution (requires matplotlib)")
    parser.add_argument("--save", type=Path, default=None,
                        help="Optional path where the final field will be stored as a NumPy file")
    return parser.parse_args(argv)


def summarise(history):
    last = history[-1]
    print("Final round-trip:")
    print(f"  number             : {last.round_trip}")
    print(f"  intracavity energy : {last.intracavity_energy:.3e} J")
    print(f"  output energy      : {last.output_energy:.3e} J")
    print(f"  NALM transmission  : {last.nalm_transmission:.3f}")
    print(f"  CW/CCW energies    : {last.cw_energy:.3e} / {last.ccw_energy:.3e} J")
    print(f"  Gain exponent      : {last.gain:.3f}")


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

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    axes[0, 0].plot(time_ps, np.abs(last_field) ** 2)
    axes[0, 0].set_xlabel("Time (ps)")
    axes[0, 0].set_ylabel("Power (W)")
    axes[0, 0].set_title("Intracavity pulse profile")

    axes[0, 1].plot(np.fft.fftshift(spectrum) * 1e-12, spectral_density)
    axes[0, 1].set_xlabel("Frequency offset (THz)")
    axes[0, 1].set_ylabel("Spectral density (a.u.)")
    axes[0, 1].set_title("Output spectrum")

    axes[1, 0].plot(round_trips, energies)
    axes[1, 0].set_xlabel("Round-trip")
    axes[1, 0].set_ylabel("Energy (J)")
    axes[1, 0].set_title("Intracavity energy evolution")

    axes[1, 1].plot(round_trips, transmissions)
    axes[1, 1].set_xlabel("Round-trip")
    axes[1, 1].set_ylabel("Transmission")
    axes[1, 1].set_title("NALM transmission")

    fig.tight_layout()
    plt.show()


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    sim = NALMFiberLaserSimulation(store_every=args.store_every)
    result = sim.run(args.round_trips, seed=args.seed, pump_bias=args.pump_bias)
    summarise(result.history)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save, result.field_history[-1])
        print(f"Saved final field to {args.save}")

    if args.plot:
        plot_results(result)


if __name__ == "__main__":
    main()
