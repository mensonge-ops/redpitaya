"""Visualization helpers for the NALM simulation results."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .simulation import SimulationResults

__all__ = ["plot_simulation_results"]


def plot_simulation_results(results: SimulationResults, *, show: bool = True, figsize: Iterable[float] = (10, 6)) -> None:
    """Reproduce the key diagnostic plots from the MATLAB workflow."""

    import matplotlib.pyplot as plt

    t = results.time
    wavelength = results.wavelength

    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(t, np.abs(results.input_field) ** 2, "b.-", label="Input")
    ax1.plot(t, np.abs(results.output_field) ** 2, "r.-", label="Output")
    ax1.set_xlabel("Time (ps)")
    ax1.set_ylabel(r"$|u(z,t)|^2$ (W)")
    ax1.set_title("Initial (blue) and output (red) pulse shapes")
    ax1.grid(True)
    ax1.legend(loc="best")

    fig2, ax2 = plt.subplots(figsize=figsize)
    ax2.plot(wavelength, results.final_spectrum_norm, "r.-")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Normalized spectrum (a.u.)")
    ax2.set_title("Output spectrum")
    ax2.grid(True)

    if results.spectral_evolution.size:
        trips = np.arange(1, results.spectral_evolution.shape[0] + 1)
        fig3, ax3 = plt.subplots(figsize=figsize)
        extent = [wavelength[0], wavelength[-1], trips.min(), trips.max()]
        im = ax3.imshow(
            results.spectral_evolution,
            aspect="auto",
            origin="lower",
            extent=extent,
        )
        ax3.set_xlabel("Wavelength (nm)")
        ax3.set_ylabel("Trip index")
        ax3.set_title("Output spectrum evolution")
        fig3.colorbar(im, ax=ax3, label="Normalized spectral power (a.u.)")

    if results.temporal_evolution.size:
        trips = np.arange(1, results.temporal_evolution.shape[0] + 1)
        fig4, ax4 = plt.subplots(figsize=figsize)
        extent = [t.min(), t.max(), trips.min(), trips.max()]
        im = ax4.imshow(
            np.abs(results.temporal_evolution) ** 2,
            aspect="auto",
            origin="lower",
            extent=extent,
        )
        ax4.set_xlabel("Time (ps)")
        ax4.set_ylabel("Trip index")
        ax4.set_title("Output pulse evolution")
        fig4.colorbar(im, ax=ax4, label="Intensity (W)")

    if results.z_axis.size:
        fig5, ax5 = plt.subplots(figsize=figsize)
        extent = [wavelength[0], wavelength[-1], results.z_axis.min(), results.z_axis.max()]
        im = ax5.imshow(
            results.combined_spectral_map,
            aspect="auto",
            origin="lower",
            extent=extent,
        )
        ax5.set_xlabel("Wavelength (nm)")
        ax5.set_ylabel("Propagation distance (km)")
        ax5.set_title("Intra-cavity spectral evolution (final trip)")
        fig5.colorbar(im, ax=ax5, label="Normalized spectral power (a.u.)")

        fig6, ax6 = plt.subplots(figsize=figsize)
        extent = [t.min(), t.max(), results.z_axis.min(), results.z_axis.max()]
        im = ax6.imshow(
            results.combined_temporal_map,
            aspect="auto",
            origin="lower",
            extent=extent,
        )
        ax6.set_xlabel("Time (ps)")
        ax6.set_ylabel("Propagation distance (km)")
        ax6.set_title("Intra-cavity pulse evolution (final trip)")
        fig6.colorbar(im, ax=ax6, label="Intensity (W)")

        colors = plt.cm.get_cmap("jet", len(results.key_labels))
        fig7, ax7 = plt.subplots(figsize=figsize)
        ax7_chirp = ax7.twinx()
        for idx, label in enumerate(results.key_labels):
            profile = results.key_profiles[label]
            color = colors(idx)
            ax7.plot(t, profile.intensity, color=color, linestyle="-", label=f"{label} intensity")
            ax7_chirp.plot(profile.chirp_time, profile.chirp, color=color, linestyle="--", label=f"{label} chirp")
        ax7.set_xlabel("Time (ps)")
        ax7.set_ylabel("Intensity (W)")
        ax7_chirp.set_ylabel("Chirp (THz)")
        ax7.set_title("Key position pulse intensity and chirp")
        ax7.grid(True)
        lines, labels = ax7.get_legend_handles_labels()
        lines2, labels2 = ax7_chirp.get_legend_handles_labels()
        ax7.legend(lines + lines2, labels + labels2, loc="best")

    if show:
        plt.show()
