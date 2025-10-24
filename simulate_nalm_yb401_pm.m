function results = simulate_nalm_yb401_pm(user_params)
%SIMULATE_NALM_YB401_PM Numerical NALM model for Yb401-PM based cavities.
%   RESULTS = SIMULATE_NALM_YB401_PM(USER_PARAMS) executes the CQEM/IP
%   solver with a cavity layout that mirrors a Yb401-PM fibre ring laser
%   terminated by a chirped fibre Bragg grating (CFBG).  The routine can
%   operate in two modes:
%
%       * Single-run mode (default) executes the simulation with the
%         provided parameters and returns the temporal/spectral evolution
%         alongside lock diagnostics.
%       * Sweep mode (set USER_PARAMS.sweep.enable = true) performs a grid
%         search across selected cavity parameters to identify combinations
%         that satisfy user-defined locking criteria (spectral bandwidth,
%         energy stability, etc.).
%
%   USER_PARAMS is optional; any fields supplied override the defaults
%   defined in DEFAULT_PARAMETERS().  Useful top-level fields are:
%       .initial.tfwhm_ps          - Seed pulse FWHM (ps)
%       .initial.N2                - Soliton order squared
%       .sections.smf_pre_L_m      - Length of the pre-gain passive fibre
%       .sections.smf_post_L_m     - Passive fibre after the gain section
%       .sections.main_gain_db     - Main loop small-signal gain (dB)
%       .sections.nalm_gain_db     - NALM gain small-signal gain (dB)
%       .coupler.rho               - NALM coupler splitting ratio
%       .coupler.rho_out           - Output coupler ratio
%       .cfbg.reflectivity         - Peak reflectivity (power)
%       .cfbg.bandwidth_nm         - FWHM bandwidth (nm)
%       .cfbg.dispersion_ps_per_nm - Group-delay slope (ps/nm)
%       .locking.min_spec_bw_nm    - Required output spectral width (nm)
%       .locking.energy_tol        - Relative energy ripple for lock detect
%       .sweep.enable              - Toggle sweep mode
%
%   RESULTS is a structure containing the simulated fields, spectra and
%   diagnostic metrics.  The field RESULTS.locking.is_locked flags whether
%   the automatic criteria were satisfied.
%
%   The helper requires the CQEM/IP utility functions shipped alongside the
%   original script (IP_CQEM_FD, coupler, gain_saturated2, etc.).
%
%   When called with no output arguments the function assigns the results
%   to the base workspace variable 'yb401_results'.

if nargin < 1
    user_params = struct();
end

params = default_parameters();
params = merge_structs(params, user_params);

if params.sweep.enable
    results = run_parameter_sweep(params);
else
    results = run_single_case(params);
end

if nargout == 0
    assignin('base', 'yb401_results', results);
end

end

%% ------------------------------------------------------------------------
function params = default_parameters()
% Baseline configuration loosely matching Nufern Yb401-PM data.
params.constants.c = 299792.458;           % nm/ps
params.initial.N2 = 1.35^2;                % soliton order squared
params.initial.tfwhm_ps = 1.8;             % ps
params.initial.lambda_nm = 1030;           % nm
params.initial.noise_level = 5e-3;         % relative amplitude noise

params.fibre.Aeff = 28.3;                  % µm^2 (6 µm MFD)
params.fibre.n2 = 26;                      % 10^-16 cm^2/W
params.fibre.loss_dB_per_m = 0.25;         % splice + absorption
params.fibre.dispersion_ps_per_nm_km = 20; % ps/(nm·km)
params.fibre.beta3_ps3_per_km = 0.12;      % ps^3/km
params.fibre.enable_raman = false;
params.fibre.enable_ss = false;

params.sections.smf_pre_L_m = 1.2;
params.sections.smf_post_L_m = 1.2;
params.sections.smf_link_L_m = 0.8;
params.sections.smf_output_L_m = 0.8;
params.sections.main_gain_L_m = 0.7;
params.sections.nalm_gain_L_m = 0.45;
params.sections.main_gain_db = 38;
params.sections.nalm_gain_db = 27;
params.sections.main_psat_dBm = 34;
params.sections.nalm_psat_dBm = 32;

params.coupler.rho = 0.45;
params.coupler.rho_out = 0.15;

params.cfbg.reflectivity = 0.18;
params.cfbg.bandwidth_nm = 20;
params.cfbg.dispersion_ps_per_nm = 0.1;

params.numerics.nt = 2^13;
params.numerics.time_window_ps = 30;
params.numerics.dz_km = 3e-6;
params.numerics.tol = 1e-4;
params.numerics.N_trip = 140;
params.numerics.quiet = true;

params.diagnostics.enable_plots = true;
params.diagnostics.plot_every = 20;
params.diagnostics.capture_segments = true;
params.diagnostics.show_waitbar = false;
params.diagnostics.focus_wavelength_nm = [1010 1050];

params.locking.window = 12;                % last round trips to analyse
params.locking.energy_tol = 0.01;          % 1 % ripple
params.locking.min_spec_bw_nm = 18;        % targeted 20 nm bandwidth
params.locking.min_peak_watts = 0.5;       % avoid trivial low-power locks

params.output.store_fields = true;
params.output.store_per_trip = true;

params.sweep.enable = false;
params.sweep.rho = 0.35:0.05:0.55;
params.sweep.rho_out = [0.1 0.15 0.2];
params.sweep.main_gain_db = 34:2:40;
params.sweep.nalm_gain_db = [24 27 30];
params.sweep.tfwhm_ps = [1.5 1.8 2.2];
params.sweep.N2 = [1.2^2 1.35^2 1.5^2];
params.sweep.max_cases = Inf;
params.sweep.rerun_best = true;
params.sweep.verbose = true;

end

%% ------------------------------------------------------------------------
function results = run_single_case(params)
% Prepare constants and derived quantities
c = params.constants.c;
lamda_pulse = params.initial.lambda_nm;
fo = c / lamda_pulse;                      % THz

% Build fibre template
fib = struct();
fib.Aeff = params.fibre.Aeff;
fib.n2 = params.fibre.n2;
fib.gamma = 2*pi*fib.n2/lamda_pulse/fib.Aeff*1e4;
fib.alpha = log(10)*params.fibre.loss_dB_per_m/10 * 1e3; % km^-1
beta2 = -(lamda_pulse^2/(2*pi*c))*params.fibre.dispersion_ps_per_nm_km;
fib.betaw = [0 0 beta2 params.fibre.beta3_ps3_per_km];
fib.raman = double(params.fibre.enable_raman);
fib.ssp = double(params.fibre.enable_ss);

% Build section copies
smf_pre = fib;  smf_pre.L = params.sections.smf_pre_L_m/1000;
smf_post = fib; smf_post.L = params.sections.smf_post_L_m/1000;
smf_link = fib; smf_link.L = params.sections.smf_link_L_m/1000;
smf_output = fib; smf_output.L = params.sections.smf_output_L_m/1000;

amf_main = fib;
amf_main.L = params.sections.main_gain_L_m/1000;
amf_main.gssdB = params.sections.main_gain_db;
amf_main.PsatdBm = params.sections.main_psat_dBm;
amf_main.lamda_gain = lamda_pulse;
amf_main.landa_bw = 7;
amf_main.fc = fo;
amf_main.fbw = c/(lamda_pulse^2) * amf_main.landa_bw;

amf_nalm = fib;
amf_nalm.L = params.sections.nalm_gain_L_m/1000;
amf_nalm.gssdB = params.sections.nalm_gain_db;
amf_nalm.PsatdBm = params.sections.nalm_psat_dBm;
amf_nalm.lamda_gain = lamda_pulse;
amf_nalm.landa_bw = 6;
amf_nalm.fc = fo;
amf_nalm.fbw = c/(lamda_pulse^2) * amf_nalm.landa_bw;

cfbg.lambda_c = lamda_pulse;
cfbg.bandwidth = params.cfbg.bandwidth_nm;
cfbg.reflectivity = params.cfbg.reflectivity;
cfbg.dispersion = params.cfbg.dispersion_ps_per_nm;
cfbg.fc = fo;
cfbg.beta2 = -cfbg.dispersion*(cfbg.lambda_c^2)/(2*pi*c);

rho = params.coupler.rho;
rho_out = params.coupler.rho_out;

% Numerical grid
nt = params.numerics.nt;
window_ps = params.numerics.time_window_ps;
dt = window_ps/nt;
t = -window_ps/2:dt:(window_ps/2-dt);

df = 1/(nt*dt);
f = -(nt/2)*df:df:(nt/2-1)*df;
lambda = c./(f + fo);

% Initial field
P_peak = 2*params.initial.N2*abs(fib.betaw(3))/fib.gamma/params.initial.tfwhm_ps^2;
u0 = sqrt(P_peak)*sech(t/params.initial.tfwhm_ps);
if params.initial.noise_level > 0
    randn('state', 0);
    noise = (1 + params.initial.noise_level*randn(1,nt));
    u0 = noise .* u0;
end

% Round-trip arrays
N_trip = params.numerics.N_trip;
if params.output.store_per_trip
    spec_z = zeros(N_trip, numel(lambda));
    u_z = zeros(N_trip, nt);
else
    spec_z = [];
    u_z = [];
end

energy_history = zeros(1, N_trip);
peak_history = zeros(1, N_trip);
spec_bw_history = zeros(1, N_trip);
pulse_fwhm_history = zeros(1, N_trip);

u = u0;
Plot_segments = struct('ufft', [], 'u', []);

if params.diagnostics.show_waitbar
    h = waitbar(0, 'Running Yb401-PM NALM simulation...');
else
    h = [];
end

for ii = 1:N_trip
    if params.diagnostics.show_waitbar
        waitbar((ii-1)/N_trip, h);
    end

    capture = params.diagnostics.capture_segments;

    [u, ~, Plot_smf_link] = IP_CQEM_FD(u, dt, params.numerics.dz_km, smf_link, fo, params.numerics.tol, capture, params.numerics.quiet);
    [u, ~, Plot_amf_main] = IP_CQEM_FD(u, dt, params.numerics.dz_km, amf_main, fo, params.numerics.tol, capture, params.numerics.quiet);
    [u, ~, Plot_smf_output] = IP_CQEM_FD(u, dt, params.numerics.dz_km, smf_output, fo, params.numerics.tol, capture, params.numerics.quiet);

    [uf, ub] = coupler(u, 0, rho);

    [uf, ~, Plot_pre_f] = IP_CQEM_FD(uf, dt, params.numerics.dz_km, smf_pre, fo, params.numerics.tol, capture, params.numerics.quiet);
    [uf, ~, Plot_gain_f] = IP_CQEM_FD(uf, dt, params.numerics.dz_km, amf_nalm, fo, params.numerics.tol, capture, params.numerics.quiet);
    [uf, ~, Plot_post_f] = IP_CQEM_FD(uf, dt, params.numerics.dz_km, smf_post, fo, params.numerics.tol, capture, params.numerics.quiet);

    [ub, ~, Plot_post_b] = IP_CQEM_FD(ub, dt, params.numerics.dz_km, smf_post, fo, params.numerics.tol, capture, params.numerics.quiet);
    [ub, ~, Plot_gain_b] = IP_CQEM_FD(ub, dt, params.numerics.dz_km, amf_nalm, fo, params.numerics.tol, capture, params.numerics.quiet);
    [ub, ~, Plot_pre_b] = IP_CQEM_FD(ub, dt, params.numerics.dz_km, smf_pre, fo, params.numerics.tol, capture, params.numerics.quiet);

    [~, ut] = coupler(ub, uf, rho);
    u = ut;

    [u, ~, Plot_smf_pre] = IP_CQEM_FD(u, dt, params.numerics.dz_km, smf_pre, fo, params.numerics.tol, capture, params.numerics.quiet);
    [u, uout] = coupler(u, 0, rho_out);

    [u, Plot_cfbg] = apply_cfbg(u, cfbg, fo, df, c, capture);

    spec = fftshift(abs(fft(uout)).^2);
    metrics = compute_spectrum_width(lambda, spec);
    spec_bw_history(ii) = metrics.fwhm_nm;

    energy = dt * sum(abs(uout).^2);
    peak_p = max(abs(uout).^2);
    energy_history(ii) = energy;
    peak_history(ii) = peak_p;

    intensity = abs(uout).^2;
    [pw_samples, ~, ~] = fwhm(intensity);
    pulse_fwhm_history(ii) = pw_samples * dt;

    if params.output.store_per_trip
        spec_z(ii, :) = spec / (max(spec) + eps);
        u_z(ii, :) = uout;
    end

    if params.diagnostics.enable_plots && (mod(ii, params.diagnostics.plot_every) == 0 || ii == 1 || ii == N_trip)
        plot_trip_progress(ii, t, u0, uout, lambda, spec, metrics, params);
    end

    if capture && ii == N_trip
        Plot_segments = concat_segments(Plot_segments, Plot_smf_link, Plot_amf_main, Plot_smf_output, ...
            Plot_pre_f, Plot_gain_f, Plot_post_f, Plot_post_b, Plot_gain_b, Plot_pre_b, Plot_smf_pre, Plot_cfbg, ut, uout);
    end
end

if ~isempty(h) && ishandle(h)
    close(h);
end

results.metrics.energy = energy_history;
results.metrics.peak_power = peak_history;
results.metrics.spec_bw_nm = spec_bw_history;
results.metrics.pulse_fwhm_ps = pulse_fwhm_history;

results.locking = evaluate_locking(params.locking, energy_history, peak_history, spec_bw_history);

results.axes.time_ps = t;
results.axes.lambda_nm = lambda;
results.parameters = params;
results.fields.u0 = u0;
results.fields.uout = uout;
results.fields.ut = ut;
results.fields.lambda = lambda;
if params.output.store_per_trip
    results.evolution.spec_z = spec_z;
    results.evolution.u_z = u_z;
else
    results.evolution = struct('spec_z', [], 'u_z', []);
end

if params.diagnostics.capture_segments
    results.plotdata = Plot_segments;
else
    results.plotdata = struct('ufft', [], 'u', []);
end

end

%% ------------------------------------------------------------------------
function sweep_results = run_parameter_sweep(params)
base_params = params;
base_params.sweep.enable = false;
base_params.diagnostics.enable_plots = false;
base_params.diagnostics.capture_segments = false;
base_params.diagnostics.show_waitbar = false;
base_params.output.store_per_trip = false;

rho_vals = params.sweep.rho;
rho_out_vals = params.sweep.rho_out;
main_gain_vals = params.sweep.main_gain_db;
nalm_gain_vals = params.sweep.nalm_gain_db;
tfwhm_vals = params.sweep.tfwhm_ps;
N2_vals = params.sweep.N2;

case_counter = 0;
best_case = [];
all_cases = struct([]);

for r = 1:numel(rho_vals)
    for ro = 1:numel(rho_out_vals)
        for gm = 1:numel(main_gain_vals)
            for gn = 1:numel(nalm_gain_vals)
                for tf = 1:numel(tfwhm_vals)
                    for nn = 1:numel(N2_vals)
                        case_counter = case_counter + 1;
                        if case_counter > params.sweep.max_cases
                            break;
                        end

                        base_params.coupler.rho = rho_vals(r);
                        base_params.coupler.rho_out = rho_out_vals(ro);
                        base_params.sections.main_gain_db = main_gain_vals(gm);
                        base_params.sections.nalm_gain_db = nalm_gain_vals(gn);
                        base_params.initial.tfwhm_ps = tfwhm_vals(tf);
                        base_params.initial.N2 = N2_vals(nn);

                        case_result = run_single_case(base_params);
                        case_result.sweep_settings = struct('rho', rho_vals(r), ...
                            'rho_out', rho_out_vals(ro), ...
                            'main_gain_db', main_gain_vals(gm), ...
                            'nalm_gain_db', nalm_gain_vals(gn), ...
                            'tfwhm_ps', tfwhm_vals(tf), ...
                            'N2', N2_vals(nn));

                        if isempty(all_cases)
                            all_cases = case_result;
                        else
                            all_cases(end+1,1) = case_result; %#ok<AGROW>
                        end

                        if params.sweep.verbose
                            fprintf('Sweep case %d: rho=%.2f rho_out=%.2f Gm=%.0f Gn=%.0f tfwhm=%.2f -> bw=%.1f nm, locked=%d\n', ...
                                case_counter, rho_vals(r), rho_out_vals(ro), main_gain_vals(gm), ...
                                nalm_gain_vals(gn), tfwhm_vals(tf), case_result.metrics.spec_bw_nm(end), ...
                                case_result.locking.is_locked);
                        end

                        if case_result.locking.is_locked
                            if isempty(best_case) || case_result.metrics.spec_bw_nm(end) > best_case.metrics.spec_bw_nm(end)
                                best_case = case_result;
                            end
                        end
                    end
                    if case_counter > params.sweep.max_cases
                        break;
                    end
                end
                if case_counter > params.sweep.max_cases
                    break;
                end
            end
            if case_counter > params.sweep.max_cases
                break;
            end
        end
        if case_counter > params.sweep.max_cases
            break;
        end
    end
    if case_counter > params.sweep.max_cases
        break;
    end
end

sweep_results.cases = all_cases;
sweep_results.best = best_case;
sweep_results.parameters = params;

if params.sweep.rerun_best && ~isempty(best_case)
    rerun_params = merge_structs(params, struct('sweep', struct('enable', false)));
    rerun_params.coupler.rho = best_case.sweep_settings.rho;
    rerun_params.coupler.rho_out = best_case.sweep_settings.rho_out;
    rerun_params.sections.main_gain_db = best_case.sweep_settings.main_gain_db;
    rerun_params.sections.nalm_gain_db = best_case.sweep_settings.nalm_gain_db;
    rerun_params.initial.tfwhm_ps = best_case.sweep_settings.tfwhm_ps;
    rerun_params.initial.N2 = best_case.sweep_settings.N2;
    rerun_params.diagnostics.enable_plots = true;
    rerun_params.diagnostics.capture_segments = true;
    rerun_params.output.store_per_trip = true;

    sweep_results.best = run_single_case(rerun_params);
    sweep_results.best.sweep_settings = best_case.sweep_settings;
end

end

%% ------------------------------------------------------------------------
function locking = evaluate_locking(criteria, energy_hist, peak_hist, spec_hist)
lock_window = min(criteria.window, numel(energy_hist));
if lock_window < 2
    lock_window = numel(energy_hist);
end

idx_range = (numel(energy_hist) - lock_window + 1):numel(energy_hist);
energies = energy_hist(idx_range);
peaks = peak_hist(idx_range);
specs = spec_hist(idx_range);

if numel(energies) < 2
    energy_ripple = 0;
else
    diffs = abs(diff(energies));
    denom = max(energies(1:end-1), eps);
    energy_ripple = max(diffs ./ denom);
end

peak_min = min(peaks);
spec_last = specs(end);

locking.energy_ripple = energy_ripple;
locking.peak_min_watts = peak_min;
locking.final_spec_bw_nm = spec_last;
locking.is_locked = (energy_ripple <= criteria.energy_tol) && ...
    (spec_last >= criteria.min_spec_bw_nm) && ...
    (peak_min >= criteria.min_peak_watts);
end

%% ------------------------------------------------------------------------
function plot_trip_progress(ii, t, u0, uout, lambda, spec, metrics, params)
figure(1); clf;
plot(t, abs(u0).^2, 'b-', t, abs(uout).^2, 'r-', 'LineWidth', 1.1);
grid on; xlabel('Time (ps)'); ylabel('|u(t)|^2 (W)');
title(sprintf('Round trip %d: temporal profile', ii));
legend('Seed', 'Output');

figure(2); clf;
plot(lambda, spec / (max(spec) + eps), 'r-', 'LineWidth', 1.1);
grid on; xlabel('Wavelength (nm)'); ylabel('Normalised spectrum');
title(sprintf('Round trip %d: spectrum (FWHM %.1f nm)', ii, metrics.fwhm_nm));
if ~isempty(params.diagnostics.focus_wavelength_nm)
    xlim(params.diagnostics.focus_wavelength_nm);
end
end

%% ------------------------------------------------------------------------
function merged = concat_segments(merged, varargin)
for k = 1:numel(varargin)
    seg = varargin{k};
    if isempty(seg)
        continue;
    end
    if isstruct(seg)
        if isfield(seg, 'ufft') && ~isempty(seg.ufft)
            merged.ufft = [merged.ufft; seg.ufft]; %#ok<AGROW>
        end
        if isfield(seg, 'u') && ~isempty(seg.u)
            merged.u = [merged.u; seg.u]; %#ok<AGROW>
        end
    else
        merged.ufft = [merged.ufft; repmat(abs(fftshift(fft(seg))), 20, 1)]; %#ok<AGROW>
        merged.u = [merged.u; repmat(seg, 20, 1)]; %#ok<AGROW>
    end
end
end

%% ------------------------------------------------------------------------
function out = merge_structs(base, update)
% Recursive struct merge utility.
if ~isstruct(update)
    out = update;
    return;
end

out = base;
fields = fieldnames(update);
for k = 1:numel(fields)
    name = fields{k};
    if isfield(base, name) && isstruct(base.(name)) && isstruct(update.(name))
        out.(name) = merge_structs(base.(name), update.(name));
    else
        out.(name) = update.(name);
    end
end
end

