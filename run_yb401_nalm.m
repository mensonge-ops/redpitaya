function [result, Plotdata] = run_yb401_nalm(params, options)
%RUN_YB401_NALM  Propagate a pulse in the Yb401-PM NALM cavity model.
%   result = RUN_YB401_NALM(params, options) executes the CQEM/IP solver
%   for the cavity described by PARAMS (see DEFAULT_YB401_PARAMS) and
%   returns a struct containing the field evolution, pulse metrics and
%   locking status.  Plotdata aggregates per-segment traces for visualisation
%   routines compatible with the legacy scripts.
%
%   Key options (all optional):
%       max_round_trips        - maximum number of cavity iterations (80)
%       lock_window            - number of recent trips used for lock check (8)
%       energy_tolerance       - relative energy variation threshold (5e-3)
%       bandwidth_tolerance_nm - allowed variation of spectral width (1.5 nm)
%       target_bandwidth_nm    - desired spectral width for convergence (20 nm)
%       minimum_bandwidth_nm   - minimum acceptable width when declaring lock
%       chirp_rms_limit        - optional limit on RMS chirp (THz)
%       adaptive_gain          - enable simple gain tuning heuristic (true)
%       gain_step_main         - main gain adjustment per trip (0.4 dB)
%       gain_step_nalm         - NALM gain adjustment per trip (0.25 dB)
%       gain_main_limits       - [min max] bounds for main gain (30 45 dB)
%       gain_nalm_limits       - [min max] bounds for NALM gain (20 34 dB)
%       collect_history        - store field/spectrum history (true)
%       collect_final_traces   - gather per-segment traces for diagnostics (true)
%       stop_on_lock           - terminate when lock criteria satisfied (true)
%       live_monitor           - plot per-trip summary (false)
%       verbose                - print status messages (true)
%
%   The function assumes all helper functions (IP_CQEM_FD, coupler, etc.)
%   are available on the MATLAB path.

if nargin < 1 || isempty(params)
    params = default_yb401_params();
end
if nargin < 2
    options = struct();
end

% ----------------------------- Options ---------------------------------
opt = options;
set_default = @(field, val) ( ~isfield(opt, field) || isempty(opt.(field)) );

if set_default('max_round_trips', [])
    opt.max_round_trips = 80;
end
if set_default('lock_window', [])
    opt.lock_window = 8;
end
if set_default('energy_tolerance', [])
    opt.energy_tolerance = 5e-3;          % relative variation
end
if set_default('bandwidth_tolerance_nm', [])
    opt.bandwidth_tolerance_nm = 1.5;
end
if set_default('target_bandwidth_nm', [])
    opt.target_bandwidth_nm = 20;
end
if set_default('minimum_bandwidth_nm', [])
    opt.minimum_bandwidth_nm = 18;
end
if set_default('chirp_rms_limit', [])
    opt.chirp_rms_limit = inf;
end
if set_default('adaptive_gain', [])
    opt.adaptive_gain = true;
end
if set_default('gain_step_main', [])
    opt.gain_step_main = 0.4;
end
if set_default('gain_step_nalm', [])
    opt.gain_step_nalm = 0.25;
end
if set_default('gain_main_limits', [])
    opt.gain_main_limits = [30 45];
end
if set_default('gain_nalm_limits', [])
    opt.gain_nalm_limits = [20 34];
end
if set_default('collect_history', [])
    opt.collect_history = true;
end
if set_default('collect_final_traces', [])
    opt.collect_final_traces = true;
end
if set_default('stop_on_lock', [])
    opt.stop_on_lock = true;
end
if set_default('live_monitor', [])
    opt.live_monitor = false;
end
if set_default('verbose', [])
    opt.verbose = true;
end

% ----------------------- Build cavity components -----------------------
c = params.constants.c;
lamda_pulse = params.pulse.lambda;
fo = c/lamda_pulse;

nt = params.grid.nt;
T_window = params.grid.time_window;
dt = T_window/nt;
t = -T_window/2:dt:(T_window/2-dt);

df = 1/(nt*dt);
f = -(nt/2)*df:df:(nt/2-1)*df;

base = struct();
base.Aeff = params.fibre.Aeff;
base.n2 = params.fibre.n2;
base.gamma = 2*pi*base.n2/lamda_pulse/base.Aeff*1e4;
base.alpha = log(10)*params.fibre.alpha_dB_per_m/10 * 1e3;  % km^-1
base.betaw = [0 0 params.fibre.beta2 params.fibre.beta3];
base.raman = double(params.fibre.include_raman);
base.ssp = double(params.fibre.include_ssp);

sections = params.sections;

smf_link = base;    smf_link.L = sections.smf_link;
smf_output = base;  smf_output.L = sections.smf_output;
smf_pre = base;     smf_pre.L = sections.smf_pre;
smf_post = base;    smf_post.L = sections.smf_post;
smf_linear = base;  smf_linear.L = sections.smf_linear;

amf_main = base;
amf_main.L = sections.amf_main.L;
amf_main.gssdB = sections.amf_main.gssdB;
amf_main.PsatdBm = sections.amf_main.PsatdBm;
amf_main.lamda_gain = lamda_pulse;
amf_main.landa_bw = sections.amf_main.bandwidth_nm;
amf_main.fc = fo;
amf_main.fbw = c/(lamda_pulse)^2 * amf_main.landa_bw;

amf_nalm = base;
amf_nalm.L = sections.amf_nalm.L;
amf_nalm.gssdB = sections.amf_nalm.gssdB;
amf_nalm.PsatdBm = sections.amf_nalm.PsatdBm;
amf_nalm.lamda_gain = lamda_pulse;
amf_nalm.landa_bw = sections.amf_nalm.bandwidth_nm;
amf_nalm.fc = fo;
amf_nalm.fbw = c/(lamda_pulse)^2 * amf_nalm.landa_bw;

cfbg = params.cfbg;
cfbg.fc = fo;
cfbg.beta2 = -cfbg.dispersion*(cfbg.lambda_c^2)/(2*pi*c);

rho = params.couplers.rho;
rho_out = params.couplers.rho_out;

dz = params.grid.dz;
tol = params.grid.tol;

% ------------------------- Initial condition ---------------------------
N2 = params.pulse.N2;
tfwhm = params.pulse.tfwhm;
P_peak = 2*N2*abs(base.betaw(3))/base.gamma/tfwhm^2;
u0 = sqrt(P_peak) * sech(t/tfwhm);

if isfield(params.pulse, 'random_seed')
    randn('state', params.pulse.random_seed);
end
if isfield(params.pulse, 'noise_level') && params.pulse.noise_level > 0
    u0 = (1 + params.pulse.noise_level*randn(1, nt)) .* u0;
end

if opt.verbose
    fprintf('\nInput peak power (W)        : %6.3f\n', max(abs(u0).^2));
    fprintf('Input pulse energy (pJ)     : %6.3f\n', dt*sum(abs(u0).^2));
    fprintf('Target spectral width (nm)  : %.2f (min %.2f)\n', ...
            opt.target_bandwidth_nm, opt.minimum_bandwidth_nm);
end

% --------------------------- Storage setup -----------------------------
max_trips = opt.max_round_trips;
u_history = [];
spec_history = [];
metrics_history(max_trips) = struct();
gain_main_history = zeros(1, max_trips);
gain_nalm_history = zeros(1, max_trips);

if opt.collect_history
    u_history = zeros(max_trips, nt);
    spec_history = zeros(max_trips, nt);
end

lock_round = NaN;
locked = false;

if opt.live_monitor
    monitor_fig = figure('Name', 'Yb401-PM NALM monitor'); %#ok<NASGU>
end

% Waitbar only when live monitoring requested to avoid GUI overhead
if opt.live_monitor
    h_wait = waitbar(0, 'Yb401-PM NALM simulation...');
end

u = u0;
final_plots = struct();

for ii = 1:max_trips
    if opt.live_monitor
        waitbar((ii-1)/max_trips, h_wait, ...
            sprintf('Trip %d/%d', ii, max_trips));
    end

    store_traces = opt.collect_final_traces;

    [u, ~, Plot_link] = IP_CQEM_FD(u, dt, dz, smf_link, fo, tol, store_traces, 0);
    [u, ~, Plot_main_gain] = IP_CQEM_FD(u, dt, dz, amf_main, fo, tol, store_traces, 0);
    [u, ~, Plot_output] = IP_CQEM_FD(u, dt, dz, smf_output, fo, tol, store_traces, 0);

    [uf, ub] = coupler(u, 0, rho);

    [uf, ~, ~] = IP_CQEM_FD(uf, dt, dz, smf_pre, fo, tol, store_traces, 0);
    [uf, ~, ~] = IP_CQEM_FD(uf, dt, dz, amf_nalm, fo, tol, store_traces, 0);
    [uf, ~, ~] = IP_CQEM_FD(uf, dt, dz, smf_post, fo, tol, store_traces, 0);

    [ub, ~, ~] = IP_CQEM_FD(ub, dt, dz, smf_post, fo, tol, store_traces, 0);
    [ub, ~, ~] = IP_CQEM_FD(ub, dt, dz, amf_nalm, fo, tol, store_traces, 0);
    [ub, ~, ~] = IP_CQEM_FD(ub, dt, dz, smf_pre, fo, tol, store_traces, 0);

    [~, ut] = coupler(ub, uf, rho);
    u = ut;

    [u, ~, Plot_pre] = IP_CQEM_FD(u, dt, dz, smf_pre, fo, tol, store_traces, 1);
    [u, uout] = coupler(u, 0, rho_out);

    [u, Plot_cfbg] = apply_cfbg(u, cfbg, fo, df, c);

    metrics = compute_locking_metrics(uout, dt, f, fo, c);
    metrics_history(ii) = metrics;
    gain_main_history(ii) = amf_main.gssdB;
    gain_nalm_history(ii) = amf_nalm.gssdB;

    if opt.collect_history
        u_history(ii, :) = uout;
        spec_history(ii, :) = metrics.spectrum;
    end

    if opt.collect_final_traces
        final_plots.Plot_link = Plot_link;
        final_plots.Plot_main_gain = Plot_main_gain;
        final_plots.Plot_output = Plot_output;
        final_plots.ut = ut;
        final_plots.Plot_pre = Plot_pre;
        final_plots.uout = uout;
        final_plots.Plot_cfbg = Plot_cfbg;
    end

    % ------------------------- Lock detection -------------------------
    window = min(opt.lock_window, ii);
    recent_metrics = metrics_history(ii-window+1:ii);
    energy_vals = [recent_metrics.energy_pJ];
    bandwidth_vals = [recent_metrics.spectral_width_nm];
    chirp_vals = [recent_metrics.chirp_rms_THz];

    energy_rel_var = max(abs(diff(energy_vals))) / max(energy_vals(end), eps);
    bandwidth_span = max(bandwidth_vals) - min(bandwidth_vals);
    mean_bandwidth = mean(bandwidth_vals);
    max_chirp = max(chirp_vals);

    if window == opt.lock_window && ...
            energy_rel_var <= opt.energy_tolerance && ...
            bandwidth_span <= opt.bandwidth_tolerance_nm && ...
            mean_bandwidth >= opt.minimum_bandwidth_nm && ...
            max_chirp <= opt.chirp_rms_limit
        locked = true;
        lock_round = ii;
        if opt.verbose
            fprintf('Lock detected at round trip %d: width = %.2f nm, energy %.3f pJ\n', ...
                ii, metrics.spectral_width_nm, metrics.energy_pJ);
        end
        if opt.stop_on_lock
            break;
        end
    end

    % --------------------- Adaptive gain heuristics -------------------
    if opt.adaptive_gain
        if metrics.spectral_width_nm < opt.target_bandwidth_nm
            amf_main.gssdB = min(amf_main.gssdB + opt.gain_step_main, opt.gain_main_limits(2));
            amf_nalm.gssdB = min(amf_nalm.gssdB + 0.5*opt.gain_step_nalm, opt.gain_nalm_limits(2));
        elseif metrics.spectral_width_nm > opt.target_bandwidth_nm + opt.bandwidth_tolerance_nm
            amf_main.gssdB = max(amf_main.gssdB - opt.gain_step_main, opt.gain_main_limits(1));
        end

        if metrics.chirp_rms_THz > opt.chirp_rms_limit && isfinite(opt.chirp_rms_limit)
            amf_nalm.gssdB = max(amf_nalm.gssdB - opt.gain_step_nalm, opt.gain_nalm_limits(1));
        end
    end

    if opt.live_monitor
        subplot(2,1,1);
        plot(t, abs(uout).^2, 'r', t, abs(u0).^2, 'k:'); grid on;
        xlabel('Time (ps)'); ylabel('|u|^2 (W)');
        title(sprintf('Trip %d intensity', ii));

        subplot(2,1,2);
        plot(metrics.lambda_axis, metrics.spectrum, 'b'); grid on;
        xlabel('Wavelength (nm)'); ylabel('Normalised spectrum');
        title(sprintf('Bandwidth %.2f nm  Gain %.1f/%.1f dB', ...
            metrics.spectral_width_nm, amf_main.gssdB, amf_nalm.gssdB));
        drawnow;
    end
end

if opt.live_monitor
    close(h_wait);
end

if isnan(lock_round)
    lock_round = max_trips;
end

% ----------------------- Assemble result struct ------------------------
result = struct();
result.u0 = u0;
result.uout = uout;
result.ut = ut;
result.time = t;
result.dt = dt;
result.f = f;
result.fo = fo;
result.c = c;
result.df = df;
result.lambda = c./(f + fo);
result.metrics_history = metrics_history(1:ii);
result.gain_main_history = gain_main_history(1:ii);
result.gain_nalm_history = gain_nalm_history(1:ii);
result.locked = locked;
result.lock_round = lock_round;
result.round_trips_executed = ii;
result.adaptive_gain_enabled = opt.adaptive_gain;
result.options = opt;

if opt.collect_history
    result.u_history = u_history(1:ii, :);
    result.spec_history = spec_history(1:ii, :);
else
    result.u_history = [];
    result.spec_history = [];
end

% Store the final metrics for convenience
result.final_metrics = metrics_history(ii);
result.final_gains = struct('amf_main', amf_main.gssdB, ...
                             'amf_nalm', amf_nalm.gssdB);

% Compose Plotdata compatible with legacy visualisations
if opt.collect_final_traces && ~isempty(fieldnames(final_plots))
    ut_fft = repmat(abs(fftshift(fft(final_plots.ut))), 20, 1);
    uout_fft = repmat(abs(fftshift(fft(final_plots.uout))), 20, 1);

    Plotdata.ufft = [final_plots.Plot_link.ufft; ...
                     final_plots.Plot_main_gain.ufft; ...
                     final_plots.Plot_output.ufft; ...
                     ut_fft; ...
                     final_plots.Plot_pre.ufft; ...
                     uout_fft; ...
                     final_plots.Plot_cfbg.ufft];

    ut_t = repmat(final_plots.ut, 20, 1);
    uout_t = repmat(final_plots.uout, 20, 1);

    Plotdata.u = [final_plots.Plot_link.u; ...
                  final_plots.Plot_main_gain.u; ...
                  final_plots.Plot_output.u; ...
                  ut_t; ...
                  final_plots.Plot_pre.u; ...
                  uout_t; ...
                  final_plots.Plot_cfbg.u];
else
    Plotdata = struct('ufft', [], 'u', []);
end

end
