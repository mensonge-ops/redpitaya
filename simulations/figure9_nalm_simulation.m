% FIGURE9_NALM_SIMULATION Simulate a figure-9 NALM mode-locked fiber laser.
%
%   This script models a traditional "figure-9" nonlinear amplifying loop
%   mirror (NALM) laser that uses a chirped fiber Bragg grating (CFBG) for
%   dispersion compensation and spectral filtering in the linear arm.  The
%   gain fiber is a 0.5 m segment of Yb401-PM fiber (600 dB/m gain coefficient
%   at 975 nm) and an additional phase shifter contributes an extra pi/2 phase
%   offset in the loop.  The total
%   optical path length corresponds to a 40 MHz repetition rate.
%
%   The simulation is implemented with a split-step Fourier method (SSFM) to
%   solve the nonlinear Schr\"odinger equation for each segment of the cavity.
%   The cavity map contains:
%       * Linear arm with a CFBG (dispersion + Gaussian filtering)
%       * 50/50 nonlinear amplifying loop mirror with a saturable Yb gain
%       * Passive fiber to satisfy a 40 MHz cavity repetition rate
%       * Output coupling for diagnostics
%
%   Key user-adjustable parameters are grouped in the PARAMS structure at the
%   top of the script, including the pump power that controls the Yb gain
%   section.  The script also produces pseudo-3D (waterfall) plots that show
%   how the temporal pulse and spectrum evolve over successive round trips
%   during the mode-locking build-up.
%
%   The script requires MATLAB R2016b or newer (for local functions).

clear; close all; clc;

%% Fundamental constants
c0 = 299792458;            % Speed of light in vacuum (m/s)
PARAMS.lambda0 = 1030e-9;  % Central wavelength (m) for Yb-doped fiber
PARAMS.omega0  = 2*pi*c0/PARAMS.lambda0;

%% Temporal grid
PARAMS.nt   = 2^12;        % Number of temporal grid points
PARAMS.Tmax = 25e-12;      % Half-width of the temporal window (s)
PARAMS.t    = linspace(-PARAMS.Tmax, PARAMS.Tmax, PARAMS.nt);
PARAMS.dt   = PARAMS.t(2) - PARAMS.t(1);
PARAMS.f    = (-PARAMS.nt/2:PARAMS.nt/2-1)/(2*PARAMS.Tmax);
PARAMS.w    = 2*pi*PARAMS.f;

%% Repetition rate and cavity length
PARAMS.Frep = 40e6;                 % Repetition frequency (Hz)
PARAMS.Trt  = 1/PARAMS.Frep;        % Round-trip time (s)
n_eff       = 1.468;                % Effective refractive index of fiber
PARAMS.Ltot = c0/(n_eff*PARAMS.Frep); % Total cavity length (m)

%% Fiber / device parameters
PARAMS.gamma_fiber   = 3.0e-3;      % Nonlinear coefficient (1/(W*m))
PARAMS.beta2_fiber   = 25e-27;      % Group velocity dispersion (s^2/m)
PARAMS.alpha_lin     = 0.05;        % Linear loss (1/m)

% Gain fiber (Yb401-PM)
PARAMS.L_gain            = 0.5;         % Length (m)
PARAMS.gain_coeff_dB     = 600;         % Small-signal gain coefficient at 975 nm (dB/m)
PARAMS.gain_coeff_linear = PARAMS.gain_coeff_dB * log(10)/10; % Convert dB/m to 1/m
PARAMS.g0_base           = 0.13 * PARAMS.gain_coeff_linear;   % Reference small-signal gain (1/m)
PARAMS.g0_max            = 0.25 * PARAMS.gain_coeff_linear;   % Clamp for small-signal gain (1/m)
PARAMS.E_sat_base        = 1.8e-6;      % Reference saturation energy (J)

% Pump configuration
PARAMS.Ppump_ref = 0.6;             % Pump power producing g0_base (W)
PARAMS.Ppump     = 1.0;             % User-selected pump power (W)

% Passive fiber to reach the target cavity length
PARAMS.L_passive = max(PARAMS.Ltot - PARAMS.L_gain, 0);

% Coupler parameters
PARAMS.kappa = 0.5;                 % Power coupling ratio (50/50)
PARAMS.T_out = 0.1;                 % Output coupling ratio (10%)

% Phase shifter
PARAMS.extra_phase = pi/2;          % Additional phase shift in the loop

% CFBG parameters
PARAMS.D_cfbg = 0.2;                % Dispersion (ps/nm)
PARAMS.BW_cfbg = 4e12;              % Spectral bandwidth (Hz) ~ 4 THz
PARAMS.R_cfbg  = 0.8;               % Peak reflectivity (unitless)

%% Simulation controls
PARAMS.Nrounds = 900;               % Number of round trips
PARAMS.noise_level = 1e-7;          % Initial complex noise amplitude
PARAMS.plot_every = 100;            % Plot interval (round trips)
PARAMS.snapshot_interval = 10;      % Round-trip spacing for pseudo-3D plots
PARAMS.verbose    = true;

%% Pre-compute operators
% Dispersion coefficient (ps/nm -> s^2)
D_si = PARAMS.D_cfbg * 1e-12 / 1e-9;               % s/m units
PARAMS.beta2_cfbg = -(PARAMS.lambda0^2/(2*pi*c0)) * D_si;  % s^2/m

% Gaussian spectral filter for the CFBG
sigma_w = PARAMS.BW_cfbg/(2*sqrt(2*log(2)));       % 1/e half-width (Hz)
PARAMS.filter_resp = PARAMS.R_cfbg * exp(-(PARAMS.w/(2*pi)).^2/(2*sigma_w^2));

% Frequency-domain dispersion operator for CFBG (single pass)
PARAMS.cfbq_phase = exp(-0.5i*PARAMS.beta2_cfbg*(PARAMS.w).^2*PARAMS.L_passive);

%% Initialize field with shot noise
A = PARAMS.noise_level * (randn(1, PARAMS.nt) + 1i*randn(1, PARAMS.nt));

%% Storage for diagnostics
pulse_energy = zeros(1, PARAMS.Nrounds);
output_field = zeros(PARAMS.Nrounds, PARAMS.nt);
snapshot_idx = 1:PARAMS.snapshot_interval:PARAMS.Nrounds;
temporal_evolution = zeros(numel(snapshot_idx), PARAMS.nt);
spectral_evolution = zeros(numel(snapshot_idx), PARAMS.nt);

%% Round-trip loop
for n = 1:PARAMS.Nrounds
    % Apply linear arm with CFBG (dispersion + filtering)
    A = apply_cfbg(A, PARAMS);

    % Coupler: split field into loop and linear arm paths
    [A_loop_in, A_lin_in] = fiber_coupler(A, zeros(size(A)), PARAMS.kappa);

    % Propagate through the nonlinear amplifying loop mirror
    A_loop_out = propagate_nalm_loop(A_loop_in, PARAMS);

    % Apply extra phase shift
    A_loop_out = A_loop_out * exp(1i*PARAMS.extra_phase);

    % Recombine at coupler (reverse propagation)
    [A_after, ~] = fiber_coupler(A_loop_out, A_lin_in, PARAMS.kappa);

    % Output coupler
    output_field(n, :) = sqrt(PARAMS.T_out) * A_after;
    A = sqrt(1 - PARAMS.T_out) * A_after;

    % Diagnostics
    pulse_energy(n) = trapz(abs(A).^2) * PARAMS.dt;

    % Store for pseudo-3D visualization
    if mod(n-1, PARAMS.snapshot_interval) == 0
        snap_pos = (n-1)/PARAMS.snapshot_interval + 1;
        temporal_evolution(snap_pos, :) = abs(A).^2;
        spectral_evolution(snap_pos, :) = abs(fftshift(fft(A))).^2;
    end

    if PARAMS.verbose && mod(n, PARAMS.plot_every) == 0
        fprintf('Round %4d: pulse energy = %.3f nJ\n', n, pulse_energy(n)*1e9);
    end
end

%% Extract final steady-state pulse
steady_field = A;
steady_spectrum = fftshift(fft(steady_field));

%% Plot results
figure('Name', 'Temporal profile');
plot(PARAMS.t*1e12, abs(steady_field).^2, 'LineWidth', 1.5);
xlabel('Time (ps)'); ylabel('Intensity (a.u.)');
title('Steady-state pulse in time domain'); grid on;

figure('Name', 'Spectral profile');
plot((PARAMS.f/1e12), abs(steady_spectrum).^2, 'LineWidth', 1.5);
xlabel('Frequency offset (THz)'); ylabel('Spectral power (a.u.)');
title('Steady-state pulse in frequency domain'); grid on;

figure('Name', 'Pulse energy evolution');
plot(1:PARAMS.Nrounds, pulse_energy*1e9, 'LineWidth', 1.5);
xlabel('Round trip number'); ylabel('Pulse energy (nJ)');
title('Pulse energy evolution'); grid on;

%% Waterfall (pseudo-3D) visualizations
t_ps = PARAMS.t * 1e12;
f_thz = PARAMS.f / 1e12;

figure('Name', 'Temporal evolution (pseudo-3D)');
waterfall(t_ps, snapshot_idx, temporal_evolution);
apply_waterfall_colormap(); shading interp; view([-35 30]);
xlabel('Time (ps)'); ylabel('Round trip'); zlabel('Intensity (a.u.)');
title('Temporal pulse evolution');

figure('Name', 'Spectral evolution (pseudo-3D)');
waterfall(f_thz, snapshot_idx, spectral_evolution);
apply_waterfall_colormap(); shading interp; view([-35 30]);
xlabel('Frequency offset (THz)'); ylabel('Round trip'); zlabel('Spectral power (a.u.)');
title('Spectral evolution');

%% Display final diagnostics
fprintf('\nFinal pulse energy: %.3f nJ\n', pulse_energy(end)*1e9);
[~, idx_peak] = max(abs(steady_field).^2);
pulse_duration = fwhm(PARAMS.t, abs(steady_field).^2);
fprintf('Estimated pulse FWHM: %.2f ps\n', pulse_duration*1e12);
fprintf('Yb gain coefficient: %.1f dB/m (%.2f 1/m)\n', ...
    PARAMS.gain_coeff_dB, PARAMS.gain_coeff_linear);
fprintf('Pump power %.2f W -> effective small-signal gain %.2f 1/m\n', ...
    PARAMS.Ppump, effective_small_signal_gain(PARAMS));

%% ----- Helper functions -----
function Aout = apply_cfbg(Ain, P)
    % Apply chirped fiber Bragg grating: dispersion + spectral filtering
    Af = fftshift(fft(Ain));
    Af = Af .* P.filter_resp .* P.cfbq_phase;
    Aout = ifft(ifftshift(Af));
end

function [E1, E2] = fiber_coupler(Ein1, Ein2, kappa)
    % Ideal lossless coupler with power splitting ratio kappa
    sqrtk = sqrt(kappa);
    sqrt1k = sqrt(1 - kappa);
    M = [sqrt1k, 1i*sqrtk; 1i*sqrtk, sqrt1k];
    fields_in = [Ein1; Ein2];
    fields_out = M * fields_in;
    E1 = fields_out(1, :);
    E2 = fields_out(2, :);
end

function Aout = propagate_nalm_loop(Ain, P)
    % Propagation through the nonlinear amplifying loop mirror
    if all(Ain == 0)
        Aout = Ain;
        return;
    end

    % Saturable gain in the gain fiber (single pass)
    g0_eff = effective_small_signal_gain(P);
    E_sat_eff = P.E_sat_base * max(P.Ppump/P.Ppump_ref, 0.1);

    A_gain = ssfm_segment(Ain, P.L_gain, P.beta2_fiber, P.gamma_fiber, P.alpha_lin, P);
    pulse_energy = trapz(abs(A_gain).^2) * P.dt;
    gain = exp(g0_eff * P.L_gain ./ (1 + pulse_energy/E_sat_eff));
    A_gain = A_gain .* gain;

    % Passive fiber propagation for the remaining loop length
    if P.L_passive > 0
        A_passive = ssfm_segment(A_gain, P.L_passive, P.beta2_fiber, P.gamma_fiber, P.alpha_lin, P);
    else
        A_passive = A_gain;
    end

    Aout = A_passive;
end

function Aout = ssfm_segment(Ain, L, beta2, gamma, alpha, P)
    % Split-step Fourier propagation for a fiber segment
    Nz = max(round(L/0.05), 1);       % At least 1 step every 5 cm
    dz = L / Nz;
    linear_op = exp((-alpha/2 - 0.5i*beta2*(P.w).^2) * dz);

    A = Ain;
    for ii = 1:Nz
        % Linear half-step
        Af = fftshift(fft(A));
        Af = Af .* sqrt(linear_op);
        A = ifft(ifftshift(Af));

        % Nonlinear step
        A = A .* exp(1i*gamma*abs(A).^2*dz);

        % Linear half-step
        Af = fftshift(fft(A));
        Af = Af .* sqrt(linear_op);
        A = ifft(ifftshift(Af));
    end
    Aout = A;
end

function width = fwhm(t, intensity)
    % Full width at half maximum helper
    intensity = intensity / max(intensity);
    halfmax = 0.5;
    idx = find(intensity >= halfmax);
    if numel(idx) < 2
        width = 0;
        return;
    end
    width = t(idx(end)) - t(idx(1));
end

function g0_eff = effective_small_signal_gain(P)
    % Pump-dependent small-signal gain coefficient for the Yb fiber
    pump_ratio = max(P.Ppump / P.Ppump_ref, 0);
    g0_eff = min(P.g0_base * pump_ratio, P.g0_max);
end

function apply_waterfall_colormap()
    % Use turbo colormap when available, otherwise fall back to parula
    if exist('turbo', 'file') == 2
        colormap(turbo);
    else
        colormap(parula);
    end
end
