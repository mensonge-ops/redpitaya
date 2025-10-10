# MATLAB SPGD Control Script

This directory contains the `spgd_control.m` utility that can run a
simultaneous perturbation stochastic gradient descent (SPGD) lock either in
simulation or against a Red Pitaya board through its SCPI interface.

## Requirements

* MATLAB R2020a or newer (for `tcpclient`, `tiledlayout`, `semilogx`, and
  `pwelch`).
* Instrument Control Toolbox for hardware mode (provides the TCP/IP client).

## Usage

Add this folder to the MATLAB path and call `spgd_control`.

```matlab
addpath('path/to/matlab');
spgd_control();                     % run the default simulation
spgd_control('Mode','hardware');    % connect to 192.168.10.2:5000 via SCPI
```

Important optional parameters include:

* `Gain` and `Perturbation` – tune the SPGD gradient step and perturbation
  amplitude.
* `Iterations` – number of iterations to execute.
* `Target` – desired intensity set-point.
* `SampleRate` – sampling rate used for the real-time PSD estimate.
* `SampleHoldTime` – dwell time after each output update before acquiring a
  sample (useful when the plant needs time to settle).
* `SimulationPlant` – struct overriding the built-in simulation model.

The live figure shows:

1. Measured intensity versus time.
2. Tracking error relative to the target intensity.
3. Single-sided intensity noise spectrum (dBc/Hz, logarithmic frequency axis)
   computed using Welch's method.
4. Control output applied to the actuator.

## Hardware Notes

* The script assumes the plant signal is present on Red Pitaya ADC channel 1
  and the actuator is driven via DAC channel 1.
* The default SCPI endpoint is `192.168.10.2:5000`; adjust using the
  `RedPitayaHost` and `RedPitayaPort` parameters.
* Ensure that the board is already configured in LV mode and that the
  necessary analog front-end connections are in place.
* The ADC query parsing logic expects responses in the default ASCII vector
  format (e.g. `{0.01,0.02,...}`).  If your firmware outputs a different
  format, adapt the `parse_red_pitaya_vector` helper inside the script.

## Simulation Model

The built-in simulation backend implements a first-order plant with a sine-wave
 disturbance and additive Gaussian noise.  Override any of the following fields
via the `SimulationPlant` parameter:

* `Gain`
* `TimeConstant`
* `DisturbanceAmplitude`
* `DisturbanceFrequency`
* `DisturbancePhase`
* `NoiseStd`
* `Offset`
* `InitialState`

Example with a slower plant:

```matlab
params = struct('TimeConstant', 5e-3, 'Gain', 1.2);
spgd_control('SimulationPlant', params, 'Iterations', 3000);
```
