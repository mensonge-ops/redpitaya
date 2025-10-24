# NALM Simulation (Python Port)

This repository now includes a pure Python reimplementation of the MATLAB
NALM simulation that ships with MMtools.  The code provides:

* A NumPy-based solver for the generalized nonlinear Schrödinger equation
  (GNLSE) using the interaction picture method with adaptive step sizing.
* Models of the optical components that form the nonlinear amplifying loop
  mirror (couplers, gain sections, filters).
* A high-level orchestration routine that reproduces the full MATLAB
  workflow and exposes the intermediate traces required for diagnostics.
* Optional plotting helpers and a command line interface for running
  experiments and exporting summaries.

## Installation

Install the runtime dependencies with your favourite package manager.  The
simulation relies on `numpy` and `matplotlib`::

```bash
pip install numpy matplotlib
```

## Running a simulation

To execute the default configuration and display the diagnostic plots run::

```bash
python examples/run_nalm_simulation.py
```

Command line options allow you to tweak the cavity parameters without touching
code.  For example, to run 10 trips and skip plot generation::

```bash
python examples/run_nalm_simulation.py --trips 10 --no-plot
```

Use `--summary path/to/file.json` to export a concise JSON report containing
runtime, pulse energy and FWHM information.

The main API entry point for library usage is
:func:`nalm.simulation.simulate_nalm`.
