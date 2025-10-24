function results = runSimulation(params)
%RUNSIMULATION Simulate a figure-of-nine NALM mode-locked fibre laser.
%
%   RESULTS = NALM.RUNSIMULATION(PARAMS) iterates the coupled rate-equation
%   and pulse-propagation model of the specified NALM cavity.  PARAMS is the
%   struct returned by nalm.defaultParameters (which can be modified prior to
%   calling this function).  The simulation uses a fourth-order Runge-Kutta
%   solver for the upper-state population dynamics and an interaction-picture
%   RK4 method for the pulse propagation inside each fibre segment.
%
%   The output RESULTS struct contains the time-domain pulses, spectra, gain
%   evolution, and diagnostic traces recorded for every round trip.
%
%   The implementation focuses on transparency and traceability rather than
%   raw execution speed.  It is suitable for MATLAB and GNU Octave.
%
%   See also: nalm.defaultParameters
%

    if nargin < 1 || isempty(params)
        params = nalm.defaultParameters();
    end

    % Prepare simulation grid and helper handles
    grid = nalm.internal.initializeGrid(params);

    % Pre-allocate diagnostic arrays
    results.time = grid.t;
    results.frequency = grid.f;
    results.round_trip_time = params.round_trip_time;
    results.params = params;

    nSteps = params.round_trips;

    results.inversion_ratio = zeros(1, nSteps);
    results.output_energy = zeros(1, nSteps);
    results.loop_energy = zeros(1, nSteps);
    results.tap_energy = zeros(1, nSteps);
    results.output_spectra = zeros(nSteps, params.Nt);
    results.output_pulses = zeros(nSteps, params.Nt);

    % Initialise population and fields
    state.N2 = params.N2_initial;
    state.linear_field = zeros(1, params.Nt);   % field returning from mirror
    state.pump_flux = params.pump_power_W ./ (params.A_eff * params.h * params.nu_pump);

    rng(params.noise_seed, 'twister');
    current_field = sqrt(params.initial_noise_W) * ...
        (randn(1, params.Nt) + 1i * randn(1, params.Nt)) / sqrt(2);

    if params.enable_plots
        figures = nalm.internal.setupPlots(grid, params);
    else
        figures = [];
    end

    for rt = 1:nSteps
        [current_field, state, metrics] = nalm.internal.roundTrip(
            current_field, state, params, grid);

        results.inversion_ratio(rt) = metrics.inversion_ratio;
        results.output_energy(rt) = metrics.output_energy;
        results.loop_energy(rt) = metrics.loop_energy;
        results.tap_energy(rt) = metrics.tap_energy;
        results.output_spectra(rt, :) = metrics.output_spectrum;
        results.output_pulses(rt, :) = metrics.output_pulse;

        if params.enable_plots && mod(rt, params.plot_every) == 0
            nalm.internal.updatePlots(figures, grid, metrics, rt);
        end
    end

    results.final_field = current_field;
    results.grid = grid;
    results.metrics_last = metrics;

    if params.enable_plots
        nalm.internal.updatePlots(figures, grid, metrics, nSteps);
    end
end
