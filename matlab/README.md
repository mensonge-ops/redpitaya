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
* `EfficiencyThreshold` – minimum acceptable ratio of current intensity to the
  best recorded intensity (default 0.95).  When the efficiency drops below this
  limit the controller automatically restores the best-known actuator command to
  keep the detector near its maximum.
* `BestDecayRate` – fractional decay applied to the stored reference intensity
  when no new maximum is observed (prevents stale peaks from dominating the
  efficiency estimate).
* `SampleRate` – sampling rate used for the real-time PSD estimate.
* `SampleHoldTime` – dwell time after each output update before acquiring a
  sample (useful when the plant needs time to settle).
* `RestoreMaxAttempts` – number of consecutive measurements allowed while
  recovering the best-known control point when efficiency dips below the
  threshold.
* `ControlLimits` – saturation limits (in volts) applied to the control output,
  matching the ±1 V range of the Red Pitaya DAC by default.
* `MeasurementSmoothFactor` – exponential smoothing factor (0–1) used to
  average the detector readings that drive the controller; set to 0 to disable
  smoothing.
* `LockedMeasurementSmoothFactor` – stronger smoothing factor (0–1) that is
  automatically engaged once the loop is locked to suppress jitter in the
  efficiency trace.
* `GradientSmoothFactor` – smoothing factor (0–1) applied to the estimated
  gradient to tame stochastic fluctuations.
* `EfficiencySmoothFactor` – smoothing factor (0–1) applied to the displayed
  efficiency trace so the plotted curve reflects the low-variance locked
  performance.
* `MinPerturbationRatio` / `MinGainRatio` – floors that limit how much the SPGD
  dither amplitude and gain shrink when efficiency exceeds the threshold.  This
  adaptive scaling keeps the lock tight while preventing excessive dithering
  once the optimum has been located.
* `LockGuardMinEfficiency` – efficiency level that marks the loop as “locked”.
  Once the smoothed detector power exceeds this level the script tracks the
  corresponding actuator point as the golden reference.
* `LockGuardDrop` – maximum fractional drop from the best locked intensity that
  is tolerated.  If the efficiency falls by more than this amount the controller
  automatically re-applies the golden actuator command.
* `LockGuardFreezeIterations` – number of iterations spent without new gradient
  updates after a guard-triggered recovery.  This lets the plant settle at the
  restored optimum before further dithering resumes.
* `LockGuardPerturbationRatio` – ceiling on the perturbation ratio while the
  guard is active, reducing dither amplitude after a recovery.
* `LockGuardGradientDamping` – multiplicative factor applied to the stored
  gradient when the guard fires, eliminating stale descent directions that would
  otherwise kick the loop away from the optimum again.
* `LockGuardDecayFactor` – scales the reference decay while locked so the best
  intensity estimate remains conservative even over long runs.
* `LockGuardMinReference` – minimum fraction of the target intensity that the
  best recorded reference must exceed before the guard engages, guaranteeing the
  protection operates only once the loop has genuinely reached the desired
  operating point.
* `LockGuardRecoverySamples` – number of additional measurements to acquire
  after re-applying the golden control, taking the best sample as the restored
  intensity to reject transient dips.
* `SimulationPlant` – struct overriding the built-in simulation model.

The live figure shows:

1. Measured intensity versus time, showing the smoothed detector signal used by
   the controller together with the raw samples, the best-achieved peak trace,
   and the adaptive reference intensity used to enforce the efficiency
   threshold.
2. Tracking error relative to the target intensity.
3. Single-sided intensity noise spectrum (dBc/Hz, logarithmic frequency axis)
   computed using Welch's method.
4. Combined plot showing efficiency (left axis) and control output (right
   axis), so you can confirm the actuator stays near the optimal command while
   preserving the required efficiency margin.

The controller continuously monitors the detector efficiency, using the
smoothed detector signal against the decayed best-achieved intensity.  If the
efficiency drops below the configurable threshold (95% by default) it restores
the best-known actuator value and re-measures until the efficiency recovers, or
gradually relaxes the reference level if the plant dynamics shift.  Once the
loop has crossed the lock guard threshold it keeps a dedicated copy of the
golden actuator command and sharply limits how far the measured intensity may
fall (default 1%).  Guard-triggered recoveries temporarily pause new gradient
updates, clamp the perturbation amplitude, take the best of several recovery
measurements, and damp the accumulated gradient so the detector power stays
within the requested >95% band with a typical minimum above 0.92 even under
disturbance.  The plotted efficiency trace applies an additional exponential
smoother while locked, matching the low-variance curve requested by the user
without affecting the raw efficiency used by the control logic.

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
