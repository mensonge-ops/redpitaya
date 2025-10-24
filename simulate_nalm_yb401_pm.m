% simulate_nalm_yb401_pm.m
% -------------------------------------------------------------------------
% High-level driver for the Yb401-PM based NALM fibre laser simulation.
% The script first runs the cavity with adaptive gain control aiming for a
% 20 nm output bandwidth.  If the heuristic does not converge, a coarse
% parameter scan is launched automatically to locate a more favourable
% configuration before re-running the detailed simulation for diagnostics.
% -------------------------------------------------------------------------

clear; clc;

%% ------------------------- Base parameters -----------------------------
params = default_yb401_params();

% Example tweaks: uncomment to bias the initial run
% params.sections.amf_main.gssdB = 40;
% params.sections.amf_nalm.gssdB = 28;
% params.pulse.tfwhm = 3.2;

%% --------------------------- Solver options ----------------------------
options = struct();
options.max_round_trips = 120;
options.lock_window = 10;
options.energy_tolerance = 3e-3;
options.bandwidth_tolerance_nm = 1.2;
options.target_bandwidth_nm = 20;
options.minimum_bandwidth_nm = 18.5;
options.chirp_rms_limit = 2.5;
options.adaptive_gain = true;
options.gain_step_main = 0.5;
options.gain_step_nalm = 0.3;
options.collect_history = true;
options.collect_final_traces = true;
options.stop_on_lock = true;
options.live_monitor = false;       % set true for per-trip plots
options.verbose = true;

%% -------------------------- Initial simulation ------------------------
[result, Plotdata] = run_yb401_nalm(params, options);

if ~result.locked
    fprintf('\nInitial configuration did not reach the target bandwidth.\n');
    fprintf('Launching coarse parameter scan to search for broader spectra...\n');

    sweep = struct();
    sweep.max_round_trips = 70;
    sweep.rho_values = 0.35:0.05:0.6;
    sweep.rho_out_values = 0.2:0.05:0.35;
    sweep.gain_main_values = 34:2:42;
    sweep.gain_nalm_values = 24:2:32;
    sweep.tfwhm_values = [3.0 3.5 4.0];
    sweep.N2_values = [1.2^2 1.35^2 1.5^2];

    scan_results = scan_yb401_nalm(params, options, sweep);
    best = scan_results.best;

    fprintf('Best candidate from scan: rho=%.2f, rho_out=%.2f, Gmain=%.1f dB, Gnalm=%.1f dB, BW=%.2f nm\n', ...
        best.params.couplers.rho, best.params.couplers.rho_out, ...
        best.params.sections.amf_main.gssdB, best.params.sections.amf_nalm.gssdB, ...
        best.spectral_width_nm);

    params = best.params;
    options.max_round_trips = 150;
    options.live_monitor = false;
    [result, Plotdata] = run_yb401_nalm(params, options);
end

%% --------------------------- Diagnostics -------------------------------
metrics = result.final_metrics;
time = result.time;
dt = result.dt;
f = result.f;
fo = result.fo;
c = result.c;
u0 = result.u0;
uout = result.uout;
lambda = result.lambda;

fprintf('\nFinal gains: main = %.2f dB, NALM = %.2f dB\n', ...
    result.final_gains.amf_main, result.final_gains.amf_nalm);
fprintf('Output bandwidth: %.2f nm\n', metrics.spectral_width_nm);
fprintf('Output energy: %.3f pJ\n', metrics.energy_pJ);

% Time-domain comparison
figure(1); clf;
plot(time, abs(u0).^2, 'b-', 'LineWidth', 1.1); hold on;
plot(time, abs(uout).^2, 'r-', 'LineWidth', 1.2); grid on; hold off;
xlabel('Time (ps)'); ylabel('|u|^2 (W)');
legend('Seed', 'Output');
title('Seed vs. output intensity');

% Spectrum
spec = fftshift(abs(fft(uout)).^2);
specnorm = spec ./ (lambda.^2);
specnorm = specnorm / max(specnorm);
figure(2); clf;
plot(lambda, specnorm, 'LineWidth', 1.2); grid on;
xlabel('Wavelength (nm)'); ylabel('Normalised spectrum (a.u.)');
title(sprintf('Output spectrum (%.2f nm FWHM)', metrics.spectral_width_nm));

% Temporal evolution across round trips
if ~isempty(result.u_history)
    figure(3); clf;
    surf(time, 1:size(result.u_history,1), abs(result.u_history).^2);
    shading interp; axis tight; colorbar;
    ylabel('Round trip'); xlabel('Time (ps)'); zlabel('|u|^2 (W)');
    title('Temporal evolution across round trips');
    view(0, 90);
end

% Spectral evolution
if ~isempty(result.spec_history)
    figure(4); clf;
    surf(lambda, 1:size(result.spec_history,1), result.spec_history);
    shading interp; axis tight; colorbar;
    ylabel('Round trip'); xlabel('Wavelength (nm)');
    zlabel('Normalised spectrum (a.u.)');
    title('Spectral evolution across round trips');
    view(0, 90);
end

% Output chirp analysis
phase_out = unwrap(angle(uout));
chirp = -diff(phase_out)/(2*pi*dt);
intensity_out = abs(uout).^2;
[width_samples, ~, ~] = fwhm(intensity_out);
figure(5); clf;
[ax, p1, p2] = plotyy(time, intensity_out, time(1:end-1), chirp, 'plot', 'plot'); %#ok<ASGLU>
set(p1, 'LineWidth', 1.4);
set(p2, 'LineWidth', 1.2, 'LineStyle', '--', 'Color', [0.2 0.2 0.7]);
xlabel(sprintf('Time (ps)   FWHM %.2f ps', width_samples*dt));
ylabel(ax(1), '|u|^2 (W)'); ylabel(ax(2), 'Chirp (THz)');
axis(ax(1), 'tight'); axis(ax(2), 'tight'); grid on;
title('Output intensity and instantaneous frequency');

% Final round-trip diagnostics using Plotdata
if ~isempty(Plotdata.ufft)
    figure(6); clf;
    surf(lambda, 1:size(Plotdata.ufft,1), ...
         (abs(Plotdata.ufft').^2) ./ (lambda'*ones(1, size(Plotdata.ufft,1))).^2);
    shading interp; axis tight; colorbar;
    xlabel('Wavelength (nm)'); ylabel('Segment index');
    zlabel('Normalised spectral power');
    title('Segment-resolved spectral evolution (final trip)');
    view(0, 90);

    figure(7); clf;
    surf(time, 1:size(Plotdata.u,1), abs(Plotdata.u).^2);
    shading interp; axis tight; colorbar;
    xlabel('Time (ps)'); ylabel('Segment index');
    zlabel('|u|^2 (W)');
    title('Segment-resolved temporal evolution (final trip)');
    view(0, 90);
end

%% ------------------------- Summary report ------------------------------
if result.locked
    fprintf('\nLock achieved at trip %d with %.2f nm bandwidth.\n', ...
        result.lock_round, metrics.spectral_width_nm);
else
    fprintf('\nNo lock detected within %d round trips.\n', result.round_trips_executed);
end

fprintf('Gain history (main/NALM) dB: %.1f -> %.1f / %.1f -> %.1f\n', ...
    result.gain_main_history(1), result.gain_main_history(end), ...
    result.gain_nalm_history(1), result.gain_nalm_history(end));

fprintf('\nUse scan_yb401_nalm or adjust params/options to explore further.\n');
