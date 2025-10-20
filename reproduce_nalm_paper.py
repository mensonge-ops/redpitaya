"""Reproduce the simulation figures from the compact NALM laser paper.

This script builds upon :mod:`nal_simulation` to replicate the numerical
investigations reported in the publication and augments them with additional
diagnostics (pulse/spectrum traces and pseudo 3D evolution plots).

The workflow loosely mirrors the methodology described in the paper:

* Configure a 1030 nm NALM cavity with Yb-doped gain fiber (YB401-PM) and a
  CFBG providing 0.2 ps/nm dispersion with 16 nm bandwidth.
* Run the NLSE + rate-equation solver using RK4IP / RK4 to obtain the
  steady-state mode-locked pulse.
* Sweep key design parameters (pump power, coupler splitting ratio, CFBG
  bandwidth, total dispersion) to study their influence on the locking regime
  and the average output power.
* Compute phase-noise and relative-intensity-noise spectra.
* Produce pseudo-3D plots with round-trip number on one axis and temporal or
  spectral coordinates on the other axis to visualise the evolution towards
  mode locking.

The script can be executed directly and writes all figures to the ``figures``
directory inside the repository. Users may tweak the parameter grids or the
number of round trips via command line options.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from simple_numeric import (
    abs_squared,
    arange,
    fft,
    fftshift,
    linspace,
    trapz,
)
from svg_plot import FillRegion, Panel, Scatter, Series, save_heatmap, save_panel_plots

from nal_simulation import (
    NALMLaser,
    build_default_nalm_laser,
    estimate_locking_state,
)


DEFAULT_OUTPUT_DIR = Path("figures")


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class BaselineResult:
    laser: NALMLaser
    final_field: List[complex]
    diagnostics: dict
    sampling_rate: float


def run_baseline_simulation(
    num_round_trips: int,
    time_window_ps: float,
    num_samples: int,
    pump_power_W: float,
) -> BaselineResult:
    laser, seed = build_default_nalm_laser(
        time_window_ps=time_window_ps,
        num_samples=num_samples,
        pump_power_W=pump_power_W,
    )
    final_field, diag = laser.evolve(
        seed,
        num_round_trips=num_round_trips,
        diagnostics=True,
        return_history=True,
    )
    sampling_rate = 1.0 / laser.dt
    return BaselineResult(laser=laser, final_field=final_field, diagnostics=diag, sampling_rate=sampling_rate)


def compute_axes(laser: NALMLaser, num_points: int) -> Tuple[List[float], List[float]]:
    time_axis_ps = [idx * laser.dt * 1e12 for idx in arange(num_points)]
    wavelength_nm = fftshift(laser._wavelength_grid(num_points))
    return time_axis_ps, wavelength_nm


def downsample_axis(values: Sequence[float], step: int) -> List[float]:
    if step <= 1:
        return list(values)
    out: List[float] = []
    for idx in range(0, len(values), step):
        chunk = values[idx : idx + step]
        if not chunk:
            continue
        out.append(sum(chunk) / len(chunk))
    return out


def downsample_matrix(
    data: Sequence[Sequence[float]], max_rows: int, max_cols: int
) -> Tuple[List[List[float]], int, int]:
    rows = len(data)
    cols = len(data[0]) if rows else 0
    if rows == 0 or cols == 0:
        return [], 1, 1
    row_step = max(1, rows // max_rows)
    col_step = max(1, cols // max_cols)
    result: List[List[float]] = []
    for row_idx in range(0, rows, row_step):
        row_block: List[float] = []
        row_end = min(row_idx + row_step, rows)
        for col_idx in range(0, cols, col_step):
            col_end = min(col_idx + col_step, cols)
            block = [data[r][c] for r in range(row_idx, row_end) for c in range(col_idx, col_end)]
            row_block.append(sum(block) / len(block))
        result.append(row_block)
    return result, row_step, col_step


def plot_energy_evolution(result: BaselineResult, output_dir: Path) -> None:
    energy = result.diagnostics.get("energy", [])
    pulse_width = result.diagnostics.get("pulse_width_ps", [])
    round_trips = list(range(1, len(energy) + 1))

    panels = [
        Panel(
            series=[Series(round_trips, energy, color="#1f77b4")],
            xlabel="Round trip",
            ylabel="Energy (arb.)",
            title="Round-trip energy evolution",
        ),
        Panel(
            series=[Series(round_trips, pulse_width, color="#ff7f0e")],
            xlabel="Round trip",
            ylabel="RMS width (ps)",
            title="Pulse width evolution",
        ),
    ]
    save_panel_plots(panels, output_dir / "fig_energy_evolution.svg", title="Energy diagnostics")


def plot_final_pulse_and_spectrum(result: BaselineResult, output_dir: Path) -> None:
    laser = result.laser
    field = result.final_field
    num_points = len(field)
    time_axis_ps, wavelength_nm = compute_axes(laser, num_points)

    pulse_intensity = abs_squared(field)
    spectrum = abs_squared(fftshift(fft(field)))

    panels = [
        Panel(
            series=[Series(time_axis_ps, pulse_intensity, color="#3366cc")],
            xlabel="Time (ps)",
            ylabel="Intensity (arb.)",
            title="Mode-locked pulse",
        ),
        Panel(
            series=[Series(wavelength_nm, spectrum, color="#cc5533")],
            xlabel="Wavelength (nm)",
            ylabel="Spectral power (arb.)",
            title="Optical spectrum",
        ),
    ]
    save_panel_plots(panels, output_dir / "fig_pulse_and_spectrum.svg", title="Pulse and spectrum")


def plot_pseudo_3d_traces(result: BaselineResult, output_dir: Path) -> None:
    history = result.diagnostics.get("field_history", [])
    spectra = result.diagnostics.get("spectrum_history", [])
    if not history:
        return

    laser = result.laser
    num_rounds = len(history)
    num_points = len(history[0]) if num_rounds else 0
    if num_points == 0:
        return
    time_axis_ps, wavelength_nm = compute_axes(laser, num_points)
    temporal_intensity = [[abs(val) ** 2 for val in frame] for frame in history]
    spectral_intensity = [[abs(val) ** 2 for val in frame] for frame in spectra]

    temporal_down, row_step, col_step = downsample_matrix(temporal_intensity, 200, 400)
    spectral_down, row_step_s, col_step_s = downsample_matrix(spectral_intensity, 200, 400)

    round_down = downsample_axis(list(range(1, num_rounds + 1)), row_step)
    time_down = downsample_axis(time_axis_ps, col_step)
    wavelength_down = downsample_axis(wavelength_nm, col_step_s)
    round_down_s = downsample_axis(list(range(1, num_rounds + 1)), row_step_s)

    save_heatmap(
        x_values=time_down,
        y_values=round_down,
        matrix=temporal_down,
        path=output_dir / "fig_pseudo3d_time.svg",
        title="Temporal build-up (pseudo-3D)",
        xlabel="Time (ps)",
        ylabel="Round trip",
        colorbar_label="Intensity (arb.)",
    )

    save_heatmap(
        x_values=wavelength_down,
        y_values=round_down_s,
        matrix=spectral_down,
        path=output_dir / "fig_pseudo3d_spectrum.svg",
        title="Spectral build-up (pseudo-3D)",
        xlabel="Wavelength (nm)",
        ylabel="Round trip",
        colorbar_label="Spectral power (arb.)",
    )


def plot_noise_spectra(result: BaselineResult, output_dir: Path) -> None:
    field = result.final_field
    laser = result.laser
    freqs_phase, psd_phase = laser.phase_noise(field, result.sampling_rate)
    freqs_intensity, psd_intensity = laser.intensity_noise(field, result.sampling_rate)

    phase_series = Series(freqs_phase[1:], psd_phase[1:], color="#1f77b4")
    rin_series = Series(freqs_intensity[1:], psd_intensity[1:], color="#d62728")

    panels = [
        Panel(
            series=[phase_series],
            xlabel="Frequency (Hz)",
            ylabel="Phase noise PSD",
            title="Phase noise",
            logx=True,
            logy=True,
        ),
        Panel(
            series=[rin_series],
            xlabel="Frequency (Hz)",
            ylabel="RIN PSD",
            title="Relative intensity noise",
            logx=True,
            logy=True,
        ),
    ]
    save_panel_plots(panels, output_dir / "fig_noise_spectra.svg", title="Noise spectra")


def pump_vs_coupler_sweep(
    pump_values: Iterable[float],
    coupler_values: Iterable[float],
    time_window_ps: float,
    num_samples: int,
    num_round_trips: int,
) -> Tuple[List[List[float]], List[List[bool]]]:
    pump_values = list(pump_values)
    coupler_values = list(coupler_values)
    avg_power = [[0.0 for _ in coupler_values] for _ in pump_values]
    locking_map = [[False for _ in coupler_values] for _ in pump_values]

    for i, pump in enumerate(pump_values):
        for j, ratio in enumerate(coupler_values):
            laser, seed = build_default_nalm_laser(
                time_window_ps=time_window_ps,
                num_samples=num_samples,
                pump_power_W=pump,
                coupler_ratio=ratio,
            )
            final_field, diag = laser.evolve(seed, num_round_trips=num_round_trips, diagnostics=True)
            avg_power[i][j] = trapz(abs_squared(final_field)) / laser.time_window
            locking_map[i][j] = estimate_locking_state(diag.get("energy", []))
    return avg_power, locking_map


def plot_coupler_pump_heatmap(
    pump_values: Iterable[float],
    coupler_values: Iterable[float],
    avg_power: Sequence[Sequence[float]],
    locking_map: Sequence[Sequence[bool]],
    output_dir: Path,
) -> None:
    pump_values = list(pump_values)
    coupler_values = list(coupler_values)
    scatter_points = [
        (coupler_values[col], pump_values[row])
        for row, row_vals in enumerate(locking_map)
        for col, locked in enumerate(row_vals)
        if locked
    ]
    save_heatmap(
        x_values=coupler_values,
        y_values=pump_values,
        matrix=avg_power,
        path=output_dir / "fig_coupler_vs_pump.svg",
        title="Average output power (arb.)",
        xlabel="Coupler ratio",
        ylabel="Pump power (W)",
        colorbar_label="Power (arb.)",
        scatter=scatter_points if scatter_points else None,
    )


def cfbg_bandwidth_sweep(
    bandwidth_values: Iterable[float],
    pump_power_W: float,
    num_round_trips: int,
    time_window_ps: float,
    num_samples: int,
) -> Tuple[List[float], List[float], List[bool]]:
    bandwidth_values = list(bandwidth_values)
    avg_power = [0.0 for _ in bandwidth_values]
    locking_state = [False for _ in bandwidth_values]

    for idx, bw in enumerate(bandwidth_values):
        laser, seed = build_default_nalm_laser(
            time_window_ps=time_window_ps,
            num_samples=num_samples,
            pump_power_W=pump_power_W,
            cfbg_fwhm_nm=bw,
        )
        final_field, diag = laser.evolve(seed, num_round_trips=num_round_trips, diagnostics=True)
        avg_power[idx] = trapz(abs_squared(final_field)) / laser.time_window
        locking_state[idx] = estimate_locking_state(diag.get("energy", []))
    return bandwidth_values, avg_power, locking_state


def plot_cfbg_bandwidth(
    bandwidth_values: Sequence[float],
    avg_power: Sequence[float],
    locking: Sequence[bool],
    output_dir: Path,
) -> None:
    fills: List[FillRegion] = []
    start_idx: Optional[int] = None
    for idx, locked in enumerate(locking):
        if locked and start_idx is None:
            start_idx = idx
        elif not locked and start_idx is not None:
            fills.append(
                FillRegion(
                    x0=bandwidth_values[start_idx],
                    x1=bandwidth_values[idx - 1],
                    color="#9ccc65",
                    opacity=0.2,
                )
            )
            start_idx = None
    if start_idx is not None:
        fills.append(
            FillRegion(
                x0=bandwidth_values[start_idx],
                x1=bandwidth_values[-1],
                color="#9ccc65",
                opacity=0.2,
            )
        )

    scatter = None
    locked_points = [(bandwidth_values[i], avg_power[i]) for i, locked in enumerate(locking) if locked]
    if locked_points:
        scatter = Scatter(
            x=[pt[0] for pt in locked_points],
            y=[pt[1] for pt in locked_points],
            color="#2e7d32",
            edge="#1b5e20",
            size=4,
            label="Locked",
        )

    panel = Panel(
        series=[Series(bandwidth_values, avg_power, color="#1f77b4")],
        xlabel="CFBG FWHM (nm)",
        ylabel="Average output power (arb.)",
        title="CFBG bandwidth optimisation",
        fills=fills,
        scatter=[scatter] if scatter else [],
    )
    save_panel_plots([panel], output_dir / "fig_cfbg_bandwidth.svg")


def dispersion_sweep(
    dispersion_values: Iterable[float],
    pump_power_W: float,
    num_round_trips: int,
    time_window_ps: float,
    num_samples: int,
) -> Tuple[List[float], List[float], List[bool]]:
    dispersion_values = list(dispersion_values)
    locking = [False for _ in dispersion_values]
    pulse_widths = [0.0 for _ in dispersion_values]

    for idx, disp in enumerate(dispersion_values):
        laser, seed = build_default_nalm_laser(
            time_window_ps=time_window_ps,
            num_samples=num_samples,
            pump_power_W=pump_power_W,
            total_dispersion_ps2=disp,
        )
        final_field, diag = laser.evolve(seed, num_round_trips=num_round_trips, diagnostics=True)
        locking[idx] = estimate_locking_state(diag.get("energy", []))
        if diag.get("pulse_width_ps"):
            pulse_widths[idx] = diag["pulse_width_ps"][-1]
    return dispersion_values, pulse_widths, locking


def plot_dispersion_scan(
    dispersion_values: Sequence[float],
    pulse_widths: Sequence[float],
    locking: Sequence[bool],
    output_dir: Path,
) -> None:
    scatter = None
    locked_points = [(dispersion_values[i], pulse_widths[i]) for i, locked in enumerate(locking) if locked]
    if locked_points:
        scatter = Scatter(
            x=[pt[0] for pt in locked_points],
            y=[pt[1] for pt in locked_points],
            color="#2e7d32",
            edge="#1b5e20",
            size=4,
            label="Locked",
        )

    panel = Panel(
        series=[Series(dispersion_values, pulse_widths, color="#ff7f0e")],
        xlabel="Total dispersion (ps²)",
        ylabel="Pulse RMS width (ps)",
        title="Net dispersion versus pulse width",
        scatter=[scatter] if scatter else [],
    )
    save_panel_plots([panel], output_dir / "fig_dispersion_scan.svg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-trips", type=int, default=400, help="Round trips for the baseline simulation")
    parser.add_argument("--sweep-round-trips", type=int, default=220, help="Round trips used in parameter sweeps")
    parser.add_argument("--time-window-ps", type=float, default=80.0, help="Simulation time window (ps)")
    parser.add_argument("--samples", type=int, default=4096, help="Number of temporal samples")
    parser.add_argument("--pump", type=float, default=0.7, help="Pump power for baseline (W)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to store figures")
    parser.add_argument("--pump-steps", type=int, default=8, help="Grid points for pump sweep")
    parser.add_argument("--coupler-steps", type=int, default=8, help="Grid points for coupler sweep")
    parser.add_argument("--bandwidth-steps", type=int, default=9, help="CFBG bandwidth samples")
    parser.add_argument("--dispersion-steps", type=int, default=8, help="Dispersion samples")
    parser.add_argument("--skip-sweeps", action="store_true", help="Skip parameter sweeps to focus on baseline locking")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output)

    baseline = run_baseline_simulation(
        num_round_trips=args.round_trips,
        time_window_ps=args.time_window_ps,
        num_samples=args.samples,
        pump_power_W=args.pump,
    )

    plot_energy_evolution(baseline, args.output)
    plot_final_pulse_and_spectrum(baseline, args.output)
    plot_pseudo_3d_traces(baseline, args.output)
    plot_noise_spectra(baseline, args.output)

    if not args.skip_sweeps:
        pump_values = linspace(0.3, 1.0, args.pump_steps)
        coupler_values = linspace(0.4, 0.7, args.coupler_steps)
        avg_power, locking_map = pump_vs_coupler_sweep(
            pump_values=pump_values,
            coupler_values=coupler_values,
            time_window_ps=args.time_window_ps,
            num_samples=args.samples,
            num_round_trips=args.sweep_round_trips,
        )
        plot_coupler_pump_heatmap(pump_values, coupler_values, avg_power, locking_map, args.output)

        bandwidth_values, avg_power_bw, locking_bw = cfbg_bandwidth_sweep(
            bandwidth_values=linspace(8.0, 24.0, args.bandwidth_steps),
            pump_power_W=args.pump,
            num_round_trips=args.sweep_round_trips,
            time_window_ps=args.time_window_ps,
            num_samples=args.samples,
        )
        plot_cfbg_bandwidth(bandwidth_values, avg_power_bw, locking_bw, args.output)

        dispersion_values, pulse_widths, locking_disp = dispersion_sweep(
            dispersion_values=linspace(-0.05, 0.1, args.dispersion_steps),
            pump_power_W=args.pump,
            num_round_trips=args.sweep_round_trips,
            time_window_ps=args.time_window_ps,
            num_samples=args.samples,
        )
        plot_dispersion_scan(dispersion_values, pulse_widths, locking_disp, args.output)

    print(f"Figures stored in {args.output.resolve()}")


if __name__ == "__main__":
    main()
