"""Tools for simulating nonlinear amplifying loop mirror (NALM) cavities.

This package contains a NumPy-based port of the MATLAB toolbox that ships
with MMtools.  The modules expose utility functions for optical components,
propagation of the generalized nonlinear Schrödinger equation (GNLSE) using
an interaction-picture integrator, and a high level orchestration routine
that replicates the original MATLAB workflow while adding Pythonic
conveniences.

The main entry point is :func:`nalm.simulation.simulate_nalm`, which returns a
:class:`nalm.simulation.SimulationResults` instance containing the full set of
intermediate traces required to reproduce the plots and diagnostics from the
MATLAB version.
"""

from .simulation import (
    FiberSegment,
    GaussianFilter,
    SimulationConfig,
    SimulationResults,
    KeyProfile,
    simulate_nalm,
)
from .plotting import plot_simulation_results

__all__ = [
    "FiberSegment",
    "GaussianFilter",
    "SimulationConfig",
    "SimulationResults",
    "KeyProfile",
    "simulate_nalm",
    "plot_simulation_results",
]
