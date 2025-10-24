function results = simulate_nalm(options)
%SIMULATE_NALM Simulate a 9-shaped all-fiber NALM laser cavity.
%   RESULTS = SIMULATE_NALM(OPTIONS) runs a time-domain simulation of the
%   specified NALM configuration.  The gain fiber is the Nufern Yb401-PM with
%   parameters provided by GETYB401PMGAINPARAMS.  Rate equations are solved
%   with a classical fourth-order Runge-Kutta scheme (see SOLVE_POPULATION_RK4)
%   while optical propagation is performed with an interaction-picture RK4
%   integrator (PROPAGATE_PASSIVE_FIBER).  The function provides real-time plots
%   of the intracavity pulse and spectrum and returns a struct containing the
%   history of the output field.
%
%   OPTIONS is a struct with the following optional fields:
%       roundTrips      - number of simulated round trips (default 200)
%       samples         - number of temporal samples (power of two, default 2048)
%       timeWindow_ps   - simulation time window in picoseconds (default 200)
%       pumpPower_W     - launched pump power at 976 nm (default 1.0 W)
%       coupler_ratio   - power coupling ratio (default 0.5)
%       gamma           - nonlinear coefficient (default 2.4e-3 1/W/m)
%       dz              - propagation step (default 0.02 m)
%       plotEvery       - update plots every N round trips (default 1)
%
%   The returned struct has fields:
%       time            - time grid (s)
%       frequency       - frequency grid (Hz)
%       roundTrips      - vector of round-trip indices
%       outputHistory   - complex field at the output coupler for each round
%
%   Example:
%       results = simulate_nalm(struct('roundTrips', 400, 'plotEvery', 5));

    arguments
        options.roundTrips (1,1) double {mustBeInteger, mustBePositive} = 200
        options.samples (1,1) double {mustBeInteger, mustBePositive} = 2048
        options.timeWindow_ps (1,1) double {mustBePositive} = 200
        options.pumpPower_W (1,1) double {mustBePositive} = 1.0
        options.coupler_ratio (1,1) double {mustBePositive} = 0.5
        options.gamma (1,1) double {mustBeNonnegative} = 2.4e-3
        options.dz (1,1) double {mustBePositive} = 0.02
        options.plotEvery (1,1) double {mustBeInteger, mustBePositive} = 1
    end

    params = GetYb401PMGainParams();

    if options.coupler_ratio <= 0 || options.coupler_ratio >= 1
        error('simulate_nalm:InvalidCouplerRatio', ...
            'Coupler ratio must be between 0 and 1 (exclusive).');
    end

    dt = options.timeWindow_ps * 1e-12 / options.samples;
    t = ((0:options.samples-1).' - options.samples/2) * dt;
    df = 1 / (options.samples * dt);
    f = ((0:options.samples-1).' - options.samples/2) * df;
    w = 2*pi*f;
    f_plot = fftshift(f);

    grid = struct('t', t, 'dt', dt, 'f', f, 'w', w);

    fiber = struct('beta2', params.beta2, 'gamma', options.gamma, ...
        'alpha', 0, 'dz', options.dz);

    % Define cavity segments for the two counter-propagating paths
    passive_0p2 = 0.2;
    passive_1p2 = 1.2;
    linear_leg = 1.0;

    ringCW = {@(u) propagate_passive_fiber(u, passive_0p2, fiber, grid), ...
        @(u) propagate_gain(u, params, grid, options.pumpPower_W, fiber), ...
        @(u) propagate_passive_fiber(u, passive_1p2, fiber, grid), ...
        @(u) u * exp(-1i * pi/2), ...
        @(u) propagate_passive_fiber(u, passive_1p2, fiber, grid)};

    ringCCW = {@(u) propagate_passive_fiber(u, passive_1p2, fiber, grid), ...
        @(u) propagate_passive_fiber(u, passive_1p2, fiber, grid), ...
        @(u) propagate_gain(u, params, grid, options.pumpPower_W, fiber), ...
        @(u) propagate_passive_fiber(u, passive_0p2, fiber, grid)};

    linearArm = {@(u) propagate_passive_fiber(u, linear_leg, fiber, grid), ...
        @(u) propagate_passive_fiber(u, linear_leg, fiber, grid)};

    % Initial field: broadband noise seed
    rng(1);
    field = (randn(options.samples, 1) + 1i * randn(options.samples, 1));
    field = field / max(abs(field)) * 1e-6;

    outputHistory = zeros(options.samples, options.roundTrips);

    fig = setup_plots(t, f_plot);

    for rt = 1:options.roundTrips
        [cw, ccw] = coupler_split(field, options.coupler_ratio);
        cw = propagate_path(cw, ringCW);
        ccw = propagate_path(ccw, ringCCW);
        [field, outputField] = coupler_recombine(cw, ccw, options.coupler_ratio);
        field = propagate_path(field, linearArm);

        outputHistory(:, rt) = outputField;

        if mod(rt, options.plotEvery) == 0
            fig = update_plots(t, f_plot, outputField, field, rt, fig);
        end
    end

    results = struct('time', t, 'frequency', f_plot, 'roundTrips', 1:options.roundTrips, ...
        'outputHistory', outputHistory);
end

function out = propagate_path(field, stages)
    out = field;
    for k = 1:numel(stages)
        out = stages{k}(out);
    end
end

function [cw, ccw] = coupler_split(field, ratio)
    tau = sqrt(ratio);
    kappa = 1i * sqrt(1 - ratio);
    in1 = field;
    in2 = zeros(size(field));
    cw = tau * in1 + kappa * in2;
    ccw = kappa * in1 + tau * in2;
end

function [field, output] = coupler_recombine(cw, ccw, ratio)
    tau = sqrt(ratio);
    kappa = 1i * sqrt(1 - ratio);
    field = tau * cw + kappa * ccw;
    output = kappa * cw + tau * ccw;
end

function field = propagate_gain(field, params, grid, pumpPower, fiber)
    [~, gain_profile] = solve_population_rk4(field, params, grid, pumpPower);
    gain_profile = real(gain_profile);
    field = propagate_passive_fiber(field, params.L, fiber, grid);
    field = field .* exp(0.5 * gain_profile * params.L);
end

function fig = setup_plots(t, f)
    fig = figure('Name', 'NALM Simulation', 'NumberTitle', 'off');
    subplot(2,1,1);
    plot(t * 1e12, zeros(size(t)));
    xlabel('Time (ps)'); ylabel('Power (W)');
    title('Output Pulse'); grid on;

    subplot(2,1,2);
    plot(f * 1e-12, zeros(size(f)));
    xlabel('Frequency (THz)'); ylabel('Power (a.u.)');
    title('Output Spectrum'); grid on;
    drawnow;
end

function fig = update_plots(t, f, outputField, intracavity, rt, fig)
    if ~(ishandle(fig) && strcmp(get(fig, 'Type'), 'figure'))
        fig = setup_plots(t, f);
    else
        figure(fig);
    end

    subplot(2,1,1);
    plot(t * 1e12, abs(outputField).^2, 'LineWidth', 1.2);
    hold on;
    plot(t * 1e12, abs(intracavity).^2, '--', 'LineWidth', 1.0);
    hold off;
    xlabel('Time (ps)'); ylabel('Power (W)');
    title(sprintf('Output Pulse (Round Trip %d)', rt)); grid on;
    legend({'Output', 'Intracavity'}, 'Location', 'northeast');

    subplot(2,1,2);
    spectrum = abs(fftshift(fft(outputField))).^2;
    plot(f * 1e-12, spectrum / max(spectrum + eps), 'LineWidth', 1.2);
    xlabel('Frequency (THz)'); ylabel('Normalised Power');
    title('Output Spectrum'); grid on;
    drawnow;
end
