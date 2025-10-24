#!/usr/bin/env python3
"""Command line entry point for the NALM cavity simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from nalm import (
    SimulationConfig,
    plot_simulation_results,
    simulate_nalm,
)


def _load_config(path: Optional[Path]) -> dict:
    if path is None:
        return {}
    with Path(path).expanduser().open("r", encoding="utf8") as stream:
        return json.load(stream)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trips", type=int, default=25, help="Number of cavity round trips")
    parser.add_argument("--rho", type=float, default=0.45, help="NALM coupler splitting ratio")
    parser.add_argument("--rho-out", type=float, default=0.35, help="Output coupler splitting ratio")
    parser.add_argument("--time-window", type=float, default=70.0, help="Simulation time window (ps)")
    parser.add_argument("--tfwhm", type=float, default=50.0, help="Input pulse FWHM (ps)")
    parser.add_argument("--noise-power", type=float, default=25.0, help="Noise power in dBW for the initial field")
    parser.add_argument("--noise-seed", type=int, default=0, help="Random seed for the input noise")
    parser.add_argument("--dz", type=float, default=1e-5, help="Initial propagation step (km)")
    parser.add_argument("--tol", type=float, default=2e-4, help="Relative photon number tolerance")
    parser.add_argument("--progress", action="store_true", help="Print progress for each trip")
    parser.add_argument("--config", type=Path, help="Load additional parameters from a JSON file")
    parser.add_argument("--no-plot", action="store_true", help="Do not show diagnostic plots")
    parser.add_argument("--summary", type=Path, help="Store a JSON summary of the run")
    return parser


def summarize_results(results) -> dict:
    dt = results.dt
    energy_in = dt * np.sum(np.abs(results.input_field) ** 2)
    energy_out = dt * np.sum(np.abs(results.output_field) ** 2)
    peak_in = float(np.max(np.abs(results.input_field) ** 2))
    peak_out = float(np.max(np.abs(results.output_field) ** 2))
    return {
        "runtime": results.runtime,
        "width_ps": results.width_time,
        "energy_in_pJ": energy_in,
        "energy_out_pJ": energy_out,
        "peak_in_W": peak_in,
        "peak_out_W": peak_out,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config_kwargs = {
        "trips": args.trips,
        "rho": args.rho,
        "rho_out": args.rho_out,
        "time_window": args.time_window,
        "tfwhm": args.tfwhm,
        "noise_power_dbw": args.noise_power,
        "noise_seed": args.noise_seed,
        "dz": args.dz,
        "tol": args.tol,
        "progress": args.progress,
    }

    config_data = _load_config(args.config)
    config_kwargs.update(config_data)
    config = SimulationConfig(**config_kwargs)

    results = simulate_nalm(config)
    summary = summarize_results(results)

    print("Simulation finished in {runtime:.2f} s".format(**summary))
    print("Input energy: {energy_in_pJ:.2f} pJ".format(**summary))
    print("Output energy: {energy_out_pJ:.2f} pJ".format(**summary))
    print("Output FWHM: {width_ps:.2f} ps")

    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with Path(args.summary).open("w", encoding="utf8") as stream:
            json.dump(summary, stream, indent=2)

    if not args.no_plot:
        plot_simulation_results(results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
