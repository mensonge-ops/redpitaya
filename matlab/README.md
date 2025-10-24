# MATLAB Tools for Yb401-PM NALM Simulation

This folder contains MATLAB helpers to assemble and run an all-fiber
nonlinear amplifying loop mirror (NALM) simulation driven by the
Nufern Yb401-PM gain fiber.  The implementation focuses on gathering
material constants, solving the Yb rate equations with an RK4 scheme,
and propagating the optical field with an interaction-picture RK4
(IP-RK4) stepper.

## Contents

- `data/yb401pm_cross_sections.csv` – tabulated absorption/emission
  cross sections digitised from the Yb401-PM data sheet (900–1100 nm).
- `nalm/LoadYbCrossSections.m` – helper that loads the CSV table.
- `nalm/GetYb401PMGainParams.m` – assembles the material constants
  required by the gain model and interpolates cross sections at pump
  and signal wavelengths.
- `nalm/solve_population_rk4.m` – RK4 integrator for the excited-state
  population based on the launched pump power and instantaneous signal
  intensity.
- `nalm/propagate_passive_fiber.m` – IP-RK4 propagation routine for
  dispersive Kerr fibers.
- `nalm/simulate_nalm.m` – executable entry point that sets up the
  9-shaped NALM cavity and produces real-time plots of the output pulse
  and spectrum for every round trip.

To launch the simulation from MATLAB:

```matlab
opts = struct('roundTrips', 300, 'plotEvery', 5, 'pumpPower_W', 0.8);
results = simulate_nalm(opts);
```

The returned struct stores the temporal and spectral grids together with
the output field for each round trip so that further analysis (e.g.
Fourier filtering, autocorrelation) can be performed offline.
