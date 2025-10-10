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
    referenceControl = control;
    perturb = zeros(nActuators, 1);
    smoothedGradient = zeros(nActuators, 1);

    intensityHistory = zeros(nIter, 1);
    rawIntensityHistory = zeros(nIter, 1);
    referenceIntensityHistory = zeros(nIter, 1);
    peakIntensityHistory = zeros(nIter, 1);
    efficiencyHistory = zeros(nIter, 1);
    errorHistory = zeros(nIter, 1);
    controlHistory = zeros(nIter, nActuators);
    timeAxis = (0:nIter-1).' ./ opts.SampleRate;

    [fig, plots] = create_plots(opts, timeAxis);

    backend.apply(control);
    pause(0.05); % allow the output to settle

    referenceIntensity = -Inf;
    peakIntensity = -Inf;
    filteredIntensity = NaN;
    perturbScale = 1;
    lockActive = false;
    lockIntensity = -Inf;
    lockControl = control;
    freezeCounter = 0;
    displayEfficiency = NaN;

    for k = 1:nIter
        skipGradient = freezeCounter > 0;
        if skipGradient
            freezeCounter = freezeCounter - 1;
        else
            currentPerturb = max(opts.MinPerturbationRatio, min(1, perturbScale)) * opts.Perturbation;
            perturb(:) = currentPerturb * (2 * randi([0, 1], nActuators, 1) - 1);

            uPlus = max(min(control + perturb, opts.ControlLimits(2)), opts.ControlLimits(1));
            backend.apply(uPlus);
            pause(opts.SampleHoldTime);
            yPlus = backend.measure();

            uMinus = max(min(control - perturb, opts.ControlLimits(2)), opts.ControlLimits(1));
            backend.apply(uMinus);
            pause(opts.SampleHoldTime);
            yMinus = backend.measure();

            if ~all(isfinite([yPlus, yMinus]))
                warning('Skipping iteration %d due to invalid measurements.', k);
                controlHistory(k, :) = control;
                if k > 1
                    intensityHistory(k) = intensityHistory(k-1);
                    errorHistory(k) = errorHistory(k-1);
                    referenceIntensityHistory(k) = referenceIntensityHistory(k-1);
                    peakIntensityHistory(k) = peakIntensityHistory(k-1);
                    efficiencyHistory(k) = efficiencyHistory(k-1);
                else
                    intensityHistory(k) = NaN;
                    errorHistory(k) = NaN;
                    referenceIntensityHistory(k) = NaN;
                    peakIntensityHistory(k) = NaN;
                    efficiencyHistory(k) = NaN;
                end
                continue;
            end

            denom = uPlus - uMinus;
            zeroMask = abs(denom) < eps;
            denom(zeroMask) = eps .* sign(perturb(zeroMask) + (perturb(zeroMask) == 0));
            denom(denom == 0) = eps;
            gradient = (yPlus - yMinus) ./ denom;
            if opts.GradientSmoothFactor <= 0 || k == 1
                smoothedGradient = gradient;
            else
                smoothedGradient = (1 - opts.GradientSmoothFactor) * smoothedGradient + opts.GradientSmoothFactor * gradient;
            end

            gainScale = max(opts.MinGainRatio, min(1, perturbScale));
            control = control + (opts.Gain * gainScale) * smoothedGradient;
            control = max(min(control, opts.ControlLimits(2)), opts.ControlLimits(1));
            backend.apply(control);
        end

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
        smoothFactor = opts.MeasurementSmoothFactor;
        if lockActive && opts.LockedMeasurementSmoothFactor > opts.MeasurementSmoothFactor
            smoothFactor = opts.LockedMeasurementSmoothFactor;
        end
        if ~isfinite(filteredIntensity) || smoothFactor <= 0
            filteredIntensity = intensity;
        else
            filteredIntensity = (1 - smoothFactor) * filteredIntensity + smoothFactor * intensity;
        end

        err = opts.Target - filteredIntensity;

        peakIntensity = max(peakIntensity, filteredIntensity);

        if ~isfinite(referenceIntensity)
            referenceIntensity = filteredIntensity;
            referenceControl = control;
        else
            decay = opts.BestDecayRate;
            if lockActive
                decay = decay * opts.LockGuardDecayFactor;
            end
            referenceIntensity = referenceIntensity * (1 - decay);
            if filteredIntensity >= referenceIntensity
                referenceIntensity = filteredIntensity;
                referenceControl = control;
                if lockActive
                    lockControl = control;
                    lockIntensity = max(lockIntensity, filteredIntensity);
                end
            end
        end

        efficiency = filteredIntensity ./ max(referenceIntensity, eps);
        restored = false;
        guardTriggered = false;
        if efficiency < opts.EfficiencyThreshold && isfinite(referenceIntensity) && referenceIntensity > 0
            warning(['Efficiency %.3f below %.2f threshold at iteration %d. ', ...
                'Restoring best-known control.'], efficiency, opts.EfficiencyThreshold, k);
            recoveryControl = referenceControl;
            if lockActive
                recoveryControl = lockControl;
            end
            control = recoveryControl;
            backend.apply(control);
            pause(opts.SampleHoldTime);
            attempt = 0;
            while attempt < opts.RestoreMaxAttempts
                intensity = backend.measure();
                if ~isfinite(intensity)
                    warning('Invalid intensity during restoration attempt %d.', attempt + 1);
                    intensity = referenceIntensity;
                end
                restoreSmooth = opts.MeasurementSmoothFactor;
                if lockActive && opts.LockedMeasurementSmoothFactor > opts.MeasurementSmoothFactor
                    restoreSmooth = opts.LockedMeasurementSmoothFactor;
                end
                if ~isfinite(filteredIntensity) || restoreSmooth <= 0
                    filteredIntensity = intensity;
                else
                    filteredIntensity = (1 - restoreSmooth) * filteredIntensity + restoreSmooth * intensity;
                end
                peakIntensity = max(peakIntensity, filteredIntensity);
                if ~isfinite(referenceIntensity)
                    referenceIntensity = filteredIntensity;
                    referenceControl = control;
                else
                    decay = opts.BestDecayRate;
                    if lockActive
                        decay = decay * opts.LockGuardDecayFactor;
                    end
                    referenceIntensity = referenceIntensity * (1 - decay);
                    if filteredIntensity >= referenceIntensity
                        referenceIntensity = filteredIntensity;
                        referenceControl = control;
                    end
                end
                efficiency = filteredIntensity ./ max(referenceIntensity, eps);
                err = opts.Target - filteredIntensity;
                if efficiency >= opts.EfficiencyThreshold
                    break;
                end
                attempt = attempt + 1;
                pause(opts.SampleHoldTime);
            end
            restored = true;
        end

        if lockActive
            lockRatio = filteredIntensity ./ max(lockIntensity, eps);
            if lockRatio < 1 - opts.LockGuardDrop
                control = lockControl;
                backend.apply(control);
                recovered = -Inf;
                for jj = 1:opts.LockGuardRecoverySamples
                    pause(opts.SampleHoldTime);
                    sample = backend.measure();
                    if isfinite(sample)
                        if ~isfinite(recovered)
                            recovered = sample;
                        else
                            recovered = max(recovered, sample);
                        end
                    end
                end
                if ~isfinite(recovered)
                    warning('Invalid intensity during lock guard recovery at iteration %d.', k);
                    recovered = max(lockIntensity, referenceIntensity);
                end
                reboundSmooth = max(opts.LockedMeasurementSmoothFactor, opts.MeasurementSmoothFactor);
                if ~isfinite(filteredIntensity) || reboundSmooth <= 0
                    filteredIntensity = recovered;
                else
                    filteredIntensity = (1 - reboundSmooth) * filteredIntensity + reboundSmooth * recovered;
                end
                intensity = recovered;
                peakIntensity = max(peakIntensity, filteredIntensity);
                if ~isfinite(referenceIntensity)
                    referenceIntensity = filteredIntensity;
                    referenceControl = control;
                else
                    decay = opts.BestDecayRate * opts.LockGuardDecayFactor;
                    referenceIntensity = referenceIntensity * (1 - decay);
                    if filteredIntensity >= referenceIntensity
                        referenceIntensity = filteredIntensity;
                        referenceControl = control;
                    end
                end
                efficiency = filteredIntensity ./ max(referenceIntensity, eps);
                err = opts.Target - filteredIntensity;
                guardTriggered = true;
            end
        end

        if ~lockActive && efficiency >= opts.LockGuardMinEfficiency && isfinite(referenceIntensity) && referenceIntensity >= opts.Target * opts.LockGuardMinReference
            lockActive = true;
            lockControl = referenceControl;
            lockIntensity = max(referenceIntensity, filteredIntensity);
        elseif lockActive
            priorLock = lockIntensity;
            lockDecay = opts.BestDecayRate * opts.LockGuardDecayFactor;
            lockIntensity = priorLock * (1 - lockDecay);
            if filteredIntensity >= priorLock
                lockControl = control;
                lockIntensity = filteredIntensity;
            end
        end

        if restored || guardTriggered
            freezeCounter = max(freezeCounter, opts.LockGuardFreezeIterations);
            perturbScale = min(perturbScale, opts.LockGuardPerturbationRatio);
            smoothedGradient = smoothedGradient * opts.LockGuardGradientDamping;
            if ~lockActive
                if isfinite(referenceIntensity) && referenceIntensity >= opts.Target * opts.LockGuardMinReference
                    lockActive = true;
                    lockControl = referenceControl;
                    lockIntensity = max(referenceIntensity, filteredIntensity);
                end
            else
                lockControl = control;
                lockIntensity = max(lockIntensity, filteredIntensity);
            end
        end

        effSmooth = opts.EfficiencySmoothFactor;
        if lockActive
            effSmooth = max(effSmooth, opts.LockedMeasurementSmoothFactor);
        end
        if ~isfinite(displayEfficiency) || effSmooth <= 0
            displayEfficiency = efficiency;
        else
            displayEfficiency = (1 - effSmooth) * displayEfficiency + effSmooth * efficiency;
        end

        rawIntensityHistory(k) = intensity;
        intensityHistory(k) = filteredIntensity;
        referenceIntensityHistory(k) = referenceIntensity;
        peakIntensityHistory(k) = peakIntensity;
        efficiencyHistory(k) = displayEfficiency;
        errorHistory(k) = err;
        controlHistory(k, :) = control;

        if restored || guardTriggered
            perturbScale = min(perturbScale, opts.LockGuardPerturbationRatio);
        elseif isfinite(referenceIntensity) && referenceIntensity > 0
            drop = max(0, efficiency - opts.EfficiencyThreshold) / max(1 - opts.EfficiencyThreshold, eps);
            perturbScale = max(opts.MinPerturbationRatio, min(1, 1 - drop));
        else
            perturbScale = 1;
        end

        if mod(k, opts.PlotUpdateInterval) == 0 || k == nIter
            update_plots(plots, intensityHistory, rawIntensityHistory, ...
                referenceIntensityHistory, peakIntensityHistory, efficiencyHistory, ...
                errorHistory, controlHistory, timeAxis, opts, k);
        end

        if ~isvalid(fig)
            warning('SPGD loop stopped because the figure window was closed.');
            break;
        end
    end

    backend.apply(control); % ensure actuator is left at last value
    drawnow;

    fprintf(['\nSPGD completed %d iterations in %s mode. Final intensity %.4f ' ...
        '(raw %.4f), reference %.4f, peak %.4f (efficiency %.3f), error %.4f.\n'], ...
        k, opts.Mode, intensityHistory(k), rawIntensityHistory(k), referenceIntensity, ...
        peakIntensity, efficiencyHistory(k), errorHistory(k));
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
    addParameter(p, 'EfficiencyThreshold', 0.95, @(x) validateattributes(x, {'numeric'}, {'scalar', '>', 0, '<=', 1}));
    addParameter(p, 'BestDecayRate', 5e-4, @(x) validateattributes(x, {'numeric'}, {'scalar', '>=', 0, '<', 1}));
    addParameter(p, 'RestoreMaxAttempts', 5, @(x) validateattributes(x, {'numeric'}, {'scalar', 'integer', '>=', 1}));
    addParameter(p, 'ControlLimits', [-1, 1], @(x) validateattributes(x, {'numeric'}, {'vector', 'numel', 2, 'increasing'}));
    addParameter(p, 'MeasurementSmoothFactor', 0.35, @(x) validateattributes(x, {'numeric'}, {'scalar', '>=', 0, '<=', 1}));
    addParameter(p, 'GradientSmoothFactor', 0.2, @(x) validateattributes(x, {'numeric'}, {'scalar', '>=', 0, '<=', 1}));
    addParameter(p, 'MinPerturbationRatio', 0.05, @(x) validateattributes(x, {'numeric'}, {'scalar', '>', 0, '<=', 1}));
    addParameter(p, 'MinGainRatio', 0.05, @(x) validateattributes(x, {'numeric'}, {'scalar', '>', 0, '<=', 1}));
    addParameter(p, 'EfficiencySmoothFactor', 0.6, @(x) validateattributes(x, {'numeric'}, {'scalar', '>=', 0, '<=', 1}));
    addParameter(p, 'LockedMeasurementSmoothFactor', 0.95, @(x) validateattributes(x, {'numeric'}, {'scalar', '>=', 0, '<=', 1}));
    addParameter(p, 'LockGuardDrop', 0.01, @(x) validateattributes(x, {'numeric'}, {'scalar', '>=', 0, '<', 1}));
    addParameter(p, 'LockGuardMinEfficiency', 0.95, @(x) validateattributes(x, {'numeric'}, {'scalar', '>', 0, '<=', 1}));
    addParameter(p, 'LockGuardFreezeIterations', 80, @(x) validateattributes(x, {'numeric'}, {'scalar', 'integer', '>=', 0}));
    addParameter(p, 'LockGuardPerturbationRatio', 0.02, @(x) validateattributes(x, {'numeric'}, {'scalar', '>', 0, '<=', 1}));
    addParameter(p, 'LockGuardGradientDamping', 0.05, @(x) validateattributes(x, {'numeric'}, {'scalar', '>=', 0, '<=', 1}));
    addParameter(p, 'LockGuardDecayFactor', 0.1, @(x) validateattributes(x, {'numeric'}, {'scalar', '>=', 0, '<=', 1}));
    addParameter(p, 'LockGuardRecoverySamples', 3, @(x) validateattributes(x, {'numeric'}, {'scalar', 'integer', '>=', 1}));
    addParameter(p, 'LockGuardMinReference', 0.99, @(x) validateattributes(x, {'numeric'}, {'scalar', '>=', 0, '<=', 1}));
    parse(p, varargin{:});

    opts = p.Results;
    opts.Mode = lower(string(opts.Mode));
    opts.RedPitayaHost = char(opts.RedPitayaHost);
    opts.ControlLimits = sort(opts.ControlLimits(:).');

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
        nHold = max(1, round(opts.SampleHoldTime * opts.SampleRate));
        acc = 0;
        for ii = 1:nHold
            [sample, state] = simulate_step(state, backendState.control, plant, opts);
            acc = acc + sample;
        end
        intensity = acc / nHold;
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
    hold(plots.input.ax, 'on');
    plots.input.rawLine = plot(timeAxis, nan(size(timeAxis)), 'LineWidth', 0.8, 'Color', [0.75 0.75 0.75]);
    plots.input.line = plot(timeAxis, nan(size(timeAxis)), 'LineWidth', 1.2, 'Color', [0.0 0.45 0.74]);
    plots.input.referenceLine = plot(timeAxis, nan(size(timeAxis)), '--', 'LineWidth', 1.1, 'Color', [0.49 0.18 0.56]);
    plots.input.peakLine = plot(timeAxis, nan(size(timeAxis)), ':', 'LineWidth', 1, 'Color', [0.3 0.3 0.3]);
    hold(plots.input.ax, 'off');
    xlabel(plots.input.ax, 'Time (s)');
    ylabel(plots.input.ax, 'Intensity (arb.)');
    title(plots.input.ax, 'Measured Intensity (filtered vs. raw)');
    grid(plots.input.ax, 'on');
    lgd = legend(plots.input.ax, {'Filtered', 'Raw', 'Reference (>=95%)', 'Peak'}, 'Location', 'best');
    set(lgd, 'AutoUpdate', 'off');

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
    yyaxis(plots.output.ax, 'left');
    plots.efficiency.line = plot(timeAxis, nan(size(timeAxis)), 'LineWidth', 1.2, 'Color', [0.93 0.69 0.13]);
    ylabel(plots.output.ax, 'Efficiency');
    ylim(plots.output.ax, [0 1.05]);

    yyaxis(plots.output.ax, 'right');
    plots.output.line = plot(timeAxis, nan(size(timeAxis)), 'LineWidth', 1.2, 'Color', [0.13 0.55 0.8]);
    ylabel(plots.output.ax, 'Control (V)');

    xlabel(plots.output.ax, 'Time (s)');
    title(plots.output.ax, sprintf('Efficiency & Control (threshold %.2f)', opts.EfficiencyThreshold));
    grid(plots.output.ax, 'on');

    drawnow;
end

function update_plots(plots, intensityHistory, rawIntensityHistory, referenceIntensityHistory, ...
    peakIntensityHistory, efficiencyHistory, errorHistory, controlHistory, timeAxis, opts, k)
    idx = 1:k;
    set(plots.input.line, 'XData', timeAxis(idx), 'YData', intensityHistory(idx));
    set(plots.input.rawLine, 'XData', timeAxis(idx), 'YData', rawIntensityHistory(idx));
    set(plots.input.referenceLine, 'XData', timeAxis(idx), 'YData', referenceIntensityHistory(idx));
    set(plots.input.peakLine, 'XData', timeAxis(idx), 'YData', peakIntensityHistory(idx));
    set(plots.error.line, 'XData', timeAxis(idx), 'YData', errorHistory(idx));
    yyaxis(plots.output.ax, 'left');
    set(plots.efficiency.line, 'XData', timeAxis(idx), 'YData', efficiencyHistory(idx));
    ylim(plots.output.ax, [0, max(1.05, max(efficiencyHistory(idx))*1.05)]);
    yyaxis(plots.output.ax, 'right');
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
