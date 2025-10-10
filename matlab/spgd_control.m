function spgd_control(varargin)
%SPGD_CONTROL Run an SPGD control loop in simulation or with a Red Pitaya.
%   SPGD_CONTROL() runs a default simulation of a single-actuator SPGD loop
%   and plots the input signal, error, intensity noise spectrum (in dBc/Hz)
%   with a logarithmic frequency axis, and the control output.
%
%   SPGD_CONTROL('Mode','hardware') connects to a Red Pitaya at the
%   configured IP address (default 192.168.10.2:5000) via SCPI and locks a
%   single channel using simultaneous perturbation stochastic gradient
%   descent.  The script assumes the fast ADC channel 1 provides the
%   intensity signal and the fast DAC channel 1 drives the actuator.
%
%   Optional name/value arguments:
%       'Mode'              - "simulation" (default) or "hardware"
%       'Iterations'        - Number of SPGD iterations (default 2000)
%       'SampleRate'        - Sampling rate used for PSD estimation (Hz)
%                             (default 20000)
%       'Gain'              - SPGD gain coefficient (default 0.08)
%       'Perturbation'      - Perturbation amplitude applied to the
%                             actuator during gradient estimation (default
%                             0.05)
%       'Target'            - Desired normalized intensity (default 1.0)
%       'NumActuators'      - Number of control outputs (default 1)
%       'PlotUpdateInterval'- Update plots every N iterations (default 10)
%       'RedPitayaHost'     - Hostname or IP of the Red Pitaya
%                             (default "192.168.10.2")
%       'RedPitayaPort'     - SCPI port (default 5000)
%       'Timeout'           - SCPI read/write timeout in seconds (default 2)
%       'Seed'              - Random seed for reproducible perturbations
%                             (default 1)
%       'SimulationPlant'   - Struct overriding the default plant
%                             parameters when running in simulation.  See
%                             LOCAL_DEFAULT_SIM_PLANT() for details.
%
%   The function shows live plots of the input intensity, tracking error,
%   single-sided intensity noise spectrum (dBc/Hz) and actuator command.
%   The noise spectrum is computed using Welch's method and referenced to
%   the squared target intensity.
%
%   Example (simulation):
%       spgd_control('Iterations', 1500, 'Gain', 0.06, 'Perturbation', 0.02);
%
%   Example (hardware):
%       spgd_control('Mode','hardware','Iterations',1200,'Gain',0.04);
%
%   NOTE: When running in hardware mode you must ensure that the Red Pitaya
%   is configured to stream the desired analog input signal on ADC1 and that
%   DAC1 is routed to your actuator.  The script configures the waveform
%   generator to produce a DC output whose value is updated each iteration.
%
%   This script is intentionally self-contained so it can be dropped into a
%   MATLAB path or executed directly from the repository.

    opts = parse_inputs(varargin{:});
    rng(opts.Seed); %#ok<RNG>

    backend = create_backend(opts);
    cleanupObj = onCleanup(@() backend.cleanup()); %#ok<NASGU>

    nIter = opts.Iterations;
    nActuators = opts.NumActuators;
    control = zeros(nActuators, 1);
    perturb = zeros(nActuators, 1);

    intensityHistory = zeros(nIter, 1);
    errorHistory = zeros(nIter, 1);
    controlHistory = zeros(nIter, nActuators);
    timeAxis = (0:nIter-1).' ./ opts.SampleRate;

    [fig, plots] = create_plots(opts, timeAxis);

    backend.apply(control);
    pause(0.05); % allow the output to settle

    for k = 1:nIter
        perturb(:) = opts.Perturbation * (2 * randi([0, 1], nActuators, 1) - 1);

        backend.apply(control + perturb);
        pause(opts.SampleHoldTime);
        yPlus = backend.measure();

        backend.apply(control - perturb);
        pause(opts.SampleHoldTime);
        yMinus = backend.measure();

        if ~all(isfinite([yPlus, yMinus]))
            warning('Skipping iteration %d due to invalid measurements.', k);
            controlHistory(k, :) = control;
            if k > 1
                intensityHistory(k) = intensityHistory(k-1);
                errorHistory(k) = errorHistory(k-1);
            else
                intensityHistory(k) = NaN;
                errorHistory(k) = NaN;
            end
            continue;
        end

        denom = 2 * perturb;
        zeroMask = abs(denom) < eps;
        denom(zeroMask) = eps .* sign(perturb(zeroMask) + (perturb(zeroMask) == 0));
        denom(denom == 0) = eps;
        gradient = (yPlus - yMinus) ./ denom;
        control = control + opts.Gain * gradient;
        backend.apply(control);

        pause(opts.SampleHoldTime);
        intensity = backend.measure();
        if ~isfinite(intensity)
            warning('Received invalid intensity measurement at iteration %d.', k);
            if k > 1
                intensity = intensityHistory(k-1);
            else
                intensity = opts.Target;
            end
        end
        err = opts.Target - intensity;

        intensityHistory(k) = intensity;
        errorHistory(k) = err;
        controlHistory(k, :) = control;

        if mod(k, opts.PlotUpdateInterval) == 0 || k == nIter
            update_plots(plots, intensityHistory, errorHistory, controlHistory, ...
                timeAxis, opts, k);
        end

        if ~isvalid(fig)
            warning('SPGD loop stopped because the figure window was closed.');
            break;
        end
    end

    backend.apply(control); % ensure actuator is left at last value
    drawnow;

    fprintf('\nSPGD completed %d iterations in %s mode. Final intensity %.4f, error %.4f.\n', ...
        k, opts.Mode, intensityHistory(k), errorHistory(k));
end

function opts = parse_inputs(varargin)
    p = inputParser;
    p.FunctionName = 'spgd_control';
    addParameter(p, 'Mode', "simulation", @(x) any(strcmpi(x, {"simulation", "hardware"})));
    addParameter(p, 'Iterations', 2000, @(x) validateattributes(x, {'numeric'}, {'scalar', 'positive', 'integer'}));
    addParameter(p, 'SampleRate', 20e3, @(x) validateattributes(x, {'numeric'}, {'scalar', 'positive'}));
    addParameter(p, 'Gain', 0.08, @(x) validateattributes(x, {'numeric'}, {'scalar', 'real'}));
    addParameter(p, 'Perturbation', 0.05, @(x) validateattributes(x, {'numeric'}, {'scalar', 'positive'}));
    addParameter(p, 'Target', 1.0, @(x) validateattributes(x, {'numeric'}, {'scalar', 'real'}));
    addParameter(p, 'NumActuators', 1, @(x) validateattributes(x, {'numeric'}, {'scalar', 'integer', '>=', 1}));
    addParameter(p, 'PlotUpdateInterval', 10, @(x) validateattributes(x, {'numeric'}, {'scalar', 'integer', '>=', 1}));
    addParameter(p, 'RedPitayaHost', "192.168.10.2", @(x) validateattributes(x, {'char', 'string'}, {'nonempty'}));
    addParameter(p, 'RedPitayaPort', 5000, @(x) validateattributes(x, {'numeric'}, {'scalar', 'integer', 'positive'}));
    addParameter(p, 'Timeout', 2.0, @(x) validateattributes(x, {'numeric'}, {'scalar', 'positive'}));
    addParameter(p, 'Seed', 1, @(x) validateattributes(x, {'numeric'}, {'scalar', 'real'}));
    addParameter(p, 'SimulationPlant', struct(), @(x) isstruct(x));
    addParameter(p, 'SampleHoldTime', 1e-3, @(x) validateattributes(x, {'numeric'}, {'scalar', 'nonnegative'}));
    parse(p, varargin{:});

    opts = p.Results;
    opts.Mode = lower(string(opts.Mode));
    opts.RedPitayaHost = char(opts.RedPitayaHost);

    if opts.NumActuators ~= 1
        warning(['The provided SPGD implementation drives a single actuator. ', ...
            'Additional actuators will share the same command value.']);
    end

    if opts.SampleHoldTime <= 0
        opts.SampleHoldTime = 1 / max(opts.SampleRate, 1);
    end

    if strcmp(opts.Mode, "simulation")
        opts.SimulationPlant = merge_struct(local_default_sim_plant(), opts.SimulationPlant);
    end
end

function backend = create_backend(opts)
    switch opts.Mode
        case "hardware"
            backend = create_red_pitaya_backend(opts);
        otherwise
            backend = create_simulation_backend(opts);
    end
end

function backend = create_simulation_backend(opts)
    plant = opts.SimulationPlant;
    state.x = plant.InitialState;
    state.t = 0;
    backendState.control = zeros(opts.NumActuators, 1);

    backend.apply = @apply_sim;
    backend.measure = @measure_sim;
    backend.cleanup = @() disp('Simulation backend shut down.');

    function apply_sim(u)
        backendState.control = u(:);
    end

    function intensity = measure_sim()
        [intensity, state] = simulate_step(state, backendState.control, plant, opts);
    end
end

function backend = create_red_pitaya_backend(opts)
    try
        client = tcpclient(opts.RedPitayaHost, opts.RedPitayaPort, 'Timeout', opts.Timeout);
    catch err
        error('Failed to connect to Red Pitaya at %s:%d (%s).', opts.RedPitayaHost, opts.RedPitayaPort, err.message);
    end

    configure_red_pitaya(client);

    backend.apply = @(u) set_red_pitaya_output(client, u);
    backend.measure = @() read_red_pitaya_adc(client);
    backend.cleanup = @() cleanup_red_pitaya(client);
end

function configure_red_pitaya(client)
    send_scpi(client, 'OUTPUT1:STATE OFF');
    send_scpi(client, 'SOUR1:FUNC DC');
    send_scpi(client, 'SOUR1:VOLT 0');
    send_scpi(client, 'OUTPUT1:STATE ON');
    send_scpi(client, 'ACQ:RST');
    send_scpi(client, 'ACQ:DEC 1');
    send_scpi(client, 'ACQ:TRIG:LEV 0');
    send_scpi(client, 'ACQ:TRIG:DLY 0');
    send_scpi(client, 'ACQ:START');
    pause(0.05);
end

function set_red_pitaya_output(client, u)
    if numel(u) > 1
        u = u(1);
    end
    cmd = sprintf('SOUR1:VOLT %0.6f', max(min(u, 1.0), -1.0));
    send_scpi(client, cmd);
end

function intensity = read_red_pitaya_adc(client)
    send_scpi(client, 'ACQ:START');
    send_scpi(client, 'ACQ:TRIG NOW');
    pause(0.002);
    writeline(client, 'ACQ:SOUR1:DATA?');
    timeout = tic;
    while client.NumBytesAvailable == 0 && toc(timeout) < 0.5
        pause(0.001);
    end
    available = client.NumBytesAvailable;
    if available == 0
        warning('No data available from Red Pitaya ADC.');
        intensity = NaN;
        return;
    end
    raw = read(client, available, 'char');
    samples = parse_red_pitaya_vector(char(raw));
    if isempty(samples)
        warning('No samples received from Red Pitaya ADC. Returning NaN.');
        intensity = NaN;
    else
        intensity = mean(samples);
    end
end

function samples = parse_red_pitaya_vector(raw)
    tokens = regexp(raw, '{([^}]*)}', 'tokens', 'once');
    if isempty(tokens)
        samples = str2double(split(raw, ','));
    else
        samples = str2double(split(tokens{1}, ','));
    end
    samples = samples(~isnan(samples));
end

function cleanup_red_pitaya(client)
    if ~isempty(client) && isvalid(client)
        try
            send_scpi(client, 'ACQ:STOP');
            send_scpi(client, 'OUTPUT1:STATE OFF');
            clear client;
        catch
        end
    end
end

function send_scpi(client, cmd)
    writeline(client, cmd);
end

function [fig, plots] = create_plots(opts, timeAxis)
    fig = figure('Name', sprintf('SPGD Control (%s)', opts.Mode), ...
        'Color', 'w', 'NumberTitle', 'off');
    t = tiledlayout(fig, 2, 2, 'Padding', 'compact');

    plots.input.ax = nexttile(t, 1);
    plots.input.line = plot(timeAxis, nan(size(timeAxis)), 'LineWidth', 1.2);
    xlabel(plots.input.ax, 'Time (s)');
    ylabel(plots.input.ax, 'Intensity (arb.)');
    title(plots.input.ax, 'Measured Intensity');
    grid(plots.input.ax, 'on');

    plots.error.ax = nexttile(t, 2);
    plots.error.line = plot(timeAxis, nan(size(timeAxis)), 'LineWidth', 1.2, 'Color', [0.85 0.33 0.1]);
    xlabel(plots.error.ax, 'Time (s)');
    ylabel(plots.error.ax, 'Error (arb.)');
    title(plots.error.ax, 'Tracking Error');
    grid(plots.error.ax, 'on');

    plots.noise.ax = nexttile(t, 3);
    plots.noise.line = semilogx([1], [nan], 'LineWidth', 1.2, 'Color', [0.47 0.67 0.19]);
    xlabel(plots.noise.ax, 'Frequency (Hz)');
    ylabel(plots.noise.ax, 'Noise (dBc/Hz)');
    title(plots.noise.ax, 'Intensity Noise Spectrum');
    grid(plots.noise.ax, 'on');

    plots.output.ax = nexttile(t, 4);
    plots.output.line = plot(timeAxis, nan(size(timeAxis)), 'LineWidth', 1.2, 'Color', [0.13 0.55 0.8]);
    xlabel(plots.output.ax, 'Time (s)');
    ylabel(plots.output.ax, 'Control (V)');
    title(plots.output.ax, 'Actuator Command');
    grid(plots.output.ax, 'on');

    drawnow;
end

function update_plots(plots, intensityHistory, errorHistory, controlHistory, timeAxis, opts, k)
    idx = 1:k;
    set(plots.input.line, 'XData', timeAxis(idx), 'YData', intensityHistory(idx));
    set(plots.error.line, 'XData', timeAxis(idx), 'YData', errorHistory(idx));
    set(plots.output.line, 'XData', timeAxis(idx), 'YData', controlHistory(idx, 1));

    validErrors = errorHistory(idx);
    validErrors = validErrors(~isnan(validErrors));
    if numel(validErrors) > 8
        [pxx, f] = pwelch(validErrors, [], [], [], opts.SampleRate);
        refPower = max(opts.Target^2, eps);
        noise = 10 * log10(max(pxx, eps) ./ refPower);
        set(plots.noise.line, 'XData', f(2:end), 'YData', noise(2:end));
        plots.noise.ax.XLimMode = 'auto';
    end

    drawnow limitrate;
end

function [intensity, state] = simulate_step(state, control, plant, opts)
    control = control(:);
    if numel(control) > 1
        control = control(1);
    end
    dt = 1 / opts.SampleRate;
    a = exp(-dt / plant.TimeConstant);
    disturbance = plant.DisturbanceAmplitude * sin(2*pi*plant.DisturbanceFrequency*state.t + plant.DisturbancePhase);
    state.x = a * state.x + (1 - a) * (plant.Gain * control + disturbance);
    state.t = state.t + dt;
    noise = plant.NoiseStd * randn();
    intensity = plant.Offset + state.x + noise;
end

function plant = local_default_sim_plant()
    plant.Gain = 0.9;
    plant.TimeConstant = 2e-3;
    plant.DisturbanceAmplitude = 0.2;
    plant.DisturbanceFrequency = 120;
    plant.DisturbancePhase = 0;
    plant.NoiseStd = 0.01;
    plant.Offset = 0.8;
    plant.InitialState = 0;
end

function merged = merge_struct(base, override)
    merged = base;
    fields = fieldnames(override);
    for k = 1:numel(fields)
        merged.(fields{k}) = override.(fields{k});
    end
end
