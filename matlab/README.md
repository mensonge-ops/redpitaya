# MATLAB NALM Laser Simulation

This folder contains a MATLAB/GNU Octave implementation of the figure-of-nine
nonlinear amplifying loop mirror (NALM) fibre laser requested by the user.  The
model combines a fourth-order Runge-Kutta rate-equation solver for the Yb401-PM
upper-state population with an interaction-picture RK4 (IP-RK4) pulse
propagation scheme for each fibre segment.  The key features are:

- **Realistic Yb401-PM data**: The `+yb/getYb401PMData.m` helper exposes
digitised absorption and emission cross sections together with lifetime, mode
area and other physical constants.  `yb.GetYbSpectrum` interpolates the spectra
at arbitrary wavelengths.
- **Full cavity layout**: `nalm.defaultParameters` sets up the ring-loop
segments (pre-coupler, 0.6 m gain fibre, WDM, phase shifter and return
sections) and the linear arm with the fibre mirror.  Pumping at 976 nm with up
to 1 W coupled power is supported.
- **Coupled solvers**: `nalm.runSimulation` performs the round-trip iteration
with RK4 rate updates and IP-RK4 pulse propagation.  Each round trip records
output pulses, spectra, loop energy and inversion.
- **Live diagnostics**: Optional real-time plots show the output pulse,
spectrum, inversion fraction and extracted energy during convergence.

To get started run the example script:

```matlab
>> demoNALMSimulation
```

For long simulations increase `params.round_trips` inside the script or modify
the struct returned by `nalm.defaultParameters` before calling
`nalm.runSimulation`.
