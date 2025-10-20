# 1030 nm NALM 9-shaped cavity fiber laser simulation plan

## Objectives

1. Demonstrate passive mode-locking behavior of a 1030 nm nonlinear amplifying loop mirror (NALM) laser with a nine-shaped cavity topology.
2. Quantify the influence of pump power (up to 1 W), round-trip loss, net cavity dispersion, gain fiber placement, and resulting average output power on the stability of the mode-locked state.
3. Extract noise characteristics, namely the phase noise spectrum and relative intensity noise (RIN), for each configuration.

## Modeling assumptions

- The field envelope propagates according to the nonlinear Schrödinger equation (NLSE) that includes second-order dispersion, Kerr nonlinearity, and distributed loss/gain. The NLSE is solved with a fourth-order Runge-Kutta interaction picture (RK4IP) method.
- The ytterbium-doped gain fiber (YB401-PM, 600 dB/m absorption at 975 nm) is modeled with a two-level rate equation solved via a fourth-order Runge-Kutta scheme. Pump absorption and signal emission cross sections are adjustable parameters.
- The chirped fiber Bragg grating (CFBG) provides 0.2 ps/nm dispersion and 16 nm FWHM spectral filtering centered at 1030 nm.
- The cavity is discretized into fiber segments (gain and passive fibers) with configurable lengths, dispersion, nonlinear coefficients, and loss.
- The nonlinear amplifying loop mirror action is captured by splitting the propagation into a loop path (gain fiber + passive fiber) and a reference arm (passive fiber) that recombine; their relative phase defines the transmission. In this simplified model we approximate the loop mirror by adjusting the effective loss and nonlinear phase imbalance at the recombination point.

## Simulation workflow

1. **Initialization**
   - Select a temporal simulation window (50–200 ps) and number of samples (4k–32k) to satisfy the Nyquist criterion for the expected pulse duration.
   - Create an initial seed field (Gaussian or noise) and normalize it to the desired initial energy.
2. **Cavity definition**
   - Define fiber segments: passive single-mode fiber sections and the Yb-doped gain fiber. Set dispersion such that the total cavity dispersion matches the user-specified value.
   - Configure lumped elements (e.g., CFBG, output couplers) as spectral-domain filters.
3. **Round-trip propagation**
   - For each segment, propagate the field using the RK4IP solver. Update the gain fiber population via the rate equations at each longitudinal step.
   - Apply lumped elements and recombine the loop mirror arms. Compute transmission as a function of nonlinear phase shift to emulate the NALM response.
4. **Mode-locking assessment**
   - Iterate for the desired number of round trips (100–500). Track energy, pulse width, and spectral bandwidth per round trip. Determine lock status by convergence of energy and pulse width metrics.
5. **Noise extraction**
   - Once steady state is achieved, compute the phase and intensity fluctuations. Use FFT-based power spectral density estimation to produce phase noise and RIN traces.
6. **Parameter sweeps**
   - Execute sweeps over pump power (0.2–1 W), distributed loss (0–3 dB), net dispersion (−0.05 to +0.1 ps²), gain fiber position (before/after CFBG), and output coupling ratios. For each sweep, record steady-state diagnostics and noise metrics.

## Data products

- Time-domain evolution plots (energy, pulse width, chirp) per round trip.
- Steady-state pulse profile and optical spectrum.
- Phase noise and intensity noise power spectral densities.
- Tables summarizing average output power and lock status versus parameter settings.
- Automated figure generation through `reproduce_nalm_paper.py` covering:
  - Pump power vs. coupler splitting ratio heatmaps indicating locking regions.
  - CFBG bandwidth sweeps versus average output power.
  - Net cavity dispersion scans highlighting achievable pulse widths.
  - Pseudo-3D temporal and spectral build-up plots for the NALM cavity.

## Implementation notes

- The provided `nal_simulation.py` module contains classes for fiber segments, gain fiber, the CFBG, and the overall NALM laser cavity, along with helper functions to run parameter sweeps and analyze lock status.
- Users can script additional experiments by importing the module and customizing segment parameters, pump power, or cavity ordering.
- The `reproduce_nalm_paper.py` script orchestrates the simulations to mirror the figures from the reference publication and saves results under `figures/` by default. Command-line options allow adjusting round trips, pump power, and sweep granularity.
