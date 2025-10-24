% simulate_nalm_yb401_pm.m
% -------------------------------------------------------------------------
% Numerical NALM mode-locking simulation configured for a cavity that
% entirely relies on Nufern (Coherent) Yb401-PM ytterbium-doped fibre.
% The implementation is derived from the reference CQEM/IP solver example
% and keeps the same numerical core while updating the fibre, gain and
% filtering parameters to reflect a realistic Yb-fibre cavity operating at
% ~1030 nm with a chirped FBG in the linear arm.
%
% Datasheet numbers for Yb401-PM (typical values, Nufern rev. K) were used
% wherever possible:
%   * 6.0 µm mode field diameter  -> Aeff ≈ 28.3 µm^2
%   * Nonlinear index n2 = 2.6e-20 m^2/W -> 26 (×10⁻¹⁶ cm²/W)
%   * Dispersion D ≈ +20 ps/(nm·km) at 1030 nm
%   * Third order dispersion ≈ 0.1 ps^3/km
%   * Small-signal absorption 6 dB/m @ 915 nm (used here as 0.25 dB/m loss)
%
% The cavity layout that is emulated here follows the ring configuration of
% the seed script: SMF spool -> gain fibre -> SMF spool -> NALM -> output
% branch -> spectral filter.  All passive sections are replaced with
% Yb401-PM so that both linear and nonlinear responses correspond to the
% ytterbium fibre.
%
% The script produces the temporal and spectral evolution through several
% cavity round trips and stores the intermediate data in Plotdata.*.
%
% -------------------------------------------------------------------------

clear; clc;

%% ------------------------------- Constants -----------------------------
c = 299792.458;                         % speed of light (nm/ps)

%% --------------------------- Input pulse -------------------------------
N2 = 1.1^2;                             % soliton order (slightly > 1 for Yb)
tfwhm = 5;                              % FWHM (ps)
lamda_pulse = 1030;                     % central wavelength (nm)
fo = c/lamda_pulse;                     % central frequency (THz)

%% ------------------------ Yb401-PM fibre core --------------------------
ybfibre.Aeff = 28.3;                    % effective area (µm^2), 6 µm MFD
% manufacturer lists n2 ≈ 2.6e-20 m^2/W -> 26 ×10^-16 cm^2/W
% use 2*pi/lambda/Aeff to get gamma in (W^-1 km^-1)
ybfibre.n2 = 26;                        % Kerr coefficient (10^-16 cm^2/W)
ybfibre.gamma = 2*pi*ybfibre.n2/lamda_pulse/ybfibre.Aeff*1e4;
% splice/coil loss is kept small (~0.25 dB/m → 0.0575 km^-1)
ybfibre.alpha = log(10)*0.25/10 * 1e3;  % convert dB/m to km^-1
% convert dispersion D=+20 ps/(nm·km) to β2=-11.26 ps^2/km, β3≈0.1 ps^3/km
ybfibre.betaw = [0 0 -11.26 0.1];
% Turn off Raman and self-steepening for base configuration
ybfibre.raman = 0;
ybfibre.ssp = 0;

%% ---------------------- Define cavity sections -------------------------
smf_pre = ybfibre;                      % pre-NALM passive spool
smf_pre.L = 0.0006;                     % 0.6 m

amf_nalm = ybfibre;                     % gain fibre inside NALM
amf_nalm.L = 0.00035;                   % 0.35 m active Yb fibre
amf_nalm.gssdB = 25;                    % small signal gain (dB)
amf_nalm.PsatdBm = 33;                  % saturation power (dBm)
amf_nalm.lamda_gain = lamda_pulse;      % gain centre (nm)
amf_nalm.landa_bw = 6;                  % emission bandwidth (nm)
amf_nalm.fc = c/amf_nalm.lamda_gain;    % centre frequency (THz)
amf_nalm.fbw = c/(amf_nalm.lamda_gain)^2*amf_nalm.landa_bw; % gain BW (THz)

smf_post = ybfibre;                     % fibre after gain (inside NALM)
smf_post.L = 0.00055;                   % 0.55 m

smf_link = ybfibre;                     % delivery fibre before filter
smf_link.L = 0.0004;                    % 0.4 m

smf_output = ybfibre;                   % extra passive section to outcoupler
smf_output.L = 0.0006;                  % 0.6 m

%% ------------------------ Output branch gain ---------------------------
amf_main = amf_nalm;                    % main loop gain fibre
amf_main.L = 0.0005;                    % 0.5 m active Yb fibre
amf_main.gssdB = 36;                    % boosted small-signal gain (dB)

%% --------------------- Chirped FBG in linear arm -----------------------
cfbg.lambda_c = lamda_pulse;            % central wavelength (nm)
cfbg.bandwidth = 20;                    % spectral FWHM (nm)
cfbg.reflectivity = 0.20;               % peak reflectivity (power)
cfbg.dispersion = 0.1;                  % ps/nm group delay slope
cfbg.fc = c/cfbg.lambda_c;              % central frequency (THz)
cfbg.beta2 = -cfbg.dispersion*(cfbg.lambda_c^2)/(2*pi*c); % ps^2
fprintf('CFBG: %.1f%% peak reflectivity, %.2f ps^2 GDD, %.1f nm FWHM.\n', ...
        cfbg.reflectivity*100, cfbg.beta2, cfbg.bandwidth);

%% ------------------------------ Couplers -------------------------------
rho = 0.5;                              % NALM coupler splitting ratio
rho_bounds = [0.35, 0.65];              % bounds for adaptive tuning

rho_out = 0.2;                          % output coupler reflectivity
rho_out_bounds = [0.1, 0.35];           % tuning bounds

%% --------------------------- Numerical grid ----------------------------
nt = 2^12;                              % number of temporal samples
time = 40;                              % total window (ps)
dt = time/nt;                           % temporal step
% use symmetric time vector
t = -time/2:dt:(time/2-dt);

% frequency grid
df = 1/(nt*dt);
f = -(nt/2)*df:df:(nt/2-1)*df;
lambda = c./(f + c/lamda_pulse);
w = 2*pi*f; %#ok<NASGU>

%% -------------------------- Propagation step ---------------------------
dz = 5e-6;                              % starting step size (km)
tol = 1e-4;                             % adaptive tolerance

%% ----------------------- Initial conditions ----------------------------
P_peak = 2*N2*abs(ybfibre.betaw(3))/ybfibre.gamma/tfwhm^2;
u0 = sqrt(P_peak)*sech(t/tfwhm);
randn('state', 0);                      % reproducible seed
u0 = (1 + 5e-3*randn(1,nt)).*u0;        % add weak noise to seed self-start

PeakPower = max(abs(u0).^2);
fprintf('\n----------------------------------------------\n');
fprintf('Input peak power (W)  = %6.3f\n', PeakPower);
fprintf('Input pulse energy (pJ) = %6.3f\n', dt*sum(abs(u0).^2));

%% --------------------------- Cavity round trips ------------------------
fprintf('\nStarting CQEM/IP propagation ...\n');
tic;

spec_z = [];
u_z = [];

u = u0;
N_trip = 120;                           % allow more round trips for settling
h1 = waitbar(0, 'Running Yb401-PM NALM simulation...');

target_bw_nm = 20;                      % desired spectral FWHM (nm)
tuning_start = 20;                      % start adapting after initial transient

energy_hist = zeros(1, N_trip);
peak_hist = zeros(1, N_trip);
spec_bw_hist = nan(1, N_trip);
rho_hist = zeros(1, N_trip);
rho_out_hist = zeros(1, N_trip);
gain_hist = zeros(1, N_trip);
lock_metric = nan(1, N_trip);

fprintf('Target spectral width: %.1f nm\n', target_bw_nm);

for ii = 1:N_trip
    waitbar((ii-1)/N_trip, h1);

    % Cavity order: smf_link -> amf_main -> smf_output before NALM
    [u, ~, Plot_smf_link] = IP_CQEM_FD(u, dt, dz, smf_link, fo, tol, 1, 0);
    [u, ~, Plot_amf_main] = IP_CQEM_FD(u, dt, dz, amf_main, fo, tol, 1, 0);
    [u, ~, Plot_smf_output] = IP_CQEM_FD(u, dt, dz, smf_output, fo, tol, 1, 0);

    % NALM coupler: forward/backward arms
    [uf, ub] = coupler(u, 0, rho);

    % forward arm: smf_pre -> amf_nalm -> smf_post
    [uf, ~, ~] = IP_CQEM_FD(uf, dt, dz, smf_pre, fo, tol, 1, 0);
    [uf, ~, ~] = IP_CQEM_FD(uf, dt, dz, amf_nalm, fo, tol, 1, 0);
    [uf, ~, ~] = IP_CQEM_FD(uf, dt, dz, smf_post, fo, tol, 1, 0);

    % backward arm: reverse order
    [ub, ~, ~] = IP_CQEM_FD(ub, dt, dz, smf_post, fo, tol, 1, 0);
    [ub, ~, ~] = IP_CQEM_FD(ub, dt, dz, amf_nalm, fo, tol, 1, 0);
    [ub, ~, ~] = IP_CQEM_FD(ub, dt, dz, smf_pre, fo, tol, 1, 0);

    % recombine at NALM coupler
    [~, ut] = coupler(ub, uf, rho);
    u = ut;

    % propagation to output coupler: smf_pre spool reused for clarity
    [u, ~, Plot_smf_pre] = IP_CQEM_FD(u, dt, dz, smf_pre, fo, tol, 1, 1);
    [u, uout] = coupler(u, 0, rho_out);

    % chirped fibre Bragg grating response (linear arm reflector)
    [u, Plot_cfbg] = apply_cfbg(u, cfbg, fo, df, c);

    % diagnostics
    figure(1); clf;
    plot(t, abs(u0).^2, 'b.-', t, abs(uout).^2, 'r.-'); axis tight;
    grid on;
    xlabel('Time (ps)'); ylabel('|u(z,t)|^2 (W)');
    title(sprintf('Seed (blue) vs. output (red) after trip %d', ii));

    spec = fftshift(abs(fft(uout)).^2);
    specnorm = spec ./ lambda.^2;
    specnorm = specnorm / max(specnorm + eps);

    figure(2); clf;
    plot(c./(f + fo), specnorm, 'r.-'); axis tight; grid on;
    xlabel('Wavelength (nm)'); ylabel('Normalised spectrum (a.u.)');
    title(sprintf('Output spectrum after trip %d', ii));

    spec_z = [spec_z; specnorm]; %#ok<AGROW>
    u_z = [u_z; uout]; %#ok<AGROW>

    % ------------ round-trip metrics & adaptive tuning -----------------
    energy_hist(ii) = dt*sum(abs(uout).^2);
    peak_hist(ii) = max(abs(uout).^2);
    spec_bw_hist(ii) = spectral_fwhm(specnorm, c./(f + fo));

    rho_hist(ii) = rho;
    rho_out_hist(ii) = rho_out;
    gain_hist(ii) = amf_main.gssdB;

    if ii > 1
        lock_metric(ii) = max(abs(energy_hist(ii) - energy_hist(ii-1))/energy_hist(ii-1), ...
                               abs(peak_hist(ii) - peak_hist(ii-1))/peak_hist(ii-1));
    end

    if ii >= tuning_start
        bw_now = spec_bw_hist(ii);
        if isnan(bw_now)
            bw_now = 0;
        end

        if bw_now < 0.7*target_bw_nm
            amf_main.gssdB = min(amf_main.gssdB + 0.8, 40);
            rho_out = max(rho_out - 0.01, rho_out_bounds(1));
            rho = min(rho + 0.01, rho_bounds(2));
            fprintf('Trip %d: bandwidth %.2f nm < target, increasing gain to %.2f dB, rho=%.3f, rho_out=%.3f\n', ...
                    ii, bw_now, amf_main.gssdB, rho, rho_out);
        elseif bw_now < 0.9*target_bw_nm
            amf_main.gssdB = min(amf_main.gssdB + 0.4, 40);
            rho_out = max(rho_out - 0.005, rho_out_bounds(1));
        elseif bw_now > 1.3*target_bw_nm
            amf_main.gssdB = max(amf_main.gssdB - 0.6, 30);
            rho_out = min(rho_out + 0.01, rho_out_bounds(2));
            rho = max(rho - 0.01, rho_bounds(1));
            fprintf('Trip %d: bandwidth %.2f nm > target, relaxing gain to %.2f dB, rho=%.3f, rho_out=%.3f\n', ...
                    ii, bw_now, amf_main.gssdB, rho, rho_out);
        end
    end
end

close(h1);
tx = toc;

%% ------------------------------ Diagnostics ----------------------------
fprintf('\nSimulation wall time (s) = %6.2f\n', tx);

lock_window = min(15, floor(N_trip/4));
locked = false;
if N_trip >= lock_window && lock_window > 3
    recent_energy = energy_hist(end-lock_window+1:end);
    recent_peak = peak_hist(end-lock_window+1:end);
    recent_bw = spec_bw_hist(end-lock_window+1:end);
    energy_var = (max(recent_energy) - min(recent_energy))/max(recent_energy);
    peak_var = (max(recent_peak) - min(recent_peak))/max(recent_peak);
    bw_mean = mean(recent_bw(~isnan(recent_bw)));
    bw_hit = bw_mean >= 0.95*target_bw_nm;
    locked = (energy_var < 0.03) && (peak_var < 0.05) && bw_hit;
    fprintf('Recent energy variation: %.3f, peak variation: %.3f, mean BW: %.2f nm\n', ...
            energy_var, peak_var, bw_mean);
end

if locked
    fprintf('Mode-locking criterion satisfied with %.2f nm spectral FWHM.\n', spec_bw_hist(end));
else
    fprintf('Mode-locking criterion NOT satisfied (final BW %.2f nm).\n', spec_bw_hist(end));
    fprintf('Consider extending N_trip or relaxing tolerances.\n');
end

figure(3);
surf(t, 1:N_trip, abs(u_z).^2);
shading interp; axis tight; colorbar;
ylabel('Round-trip index'); xlabel('Time (ps)'); zlabel('|u(z,t)|^2 (W)');
title('Temporal evolution across round trips');
view(0, 90);

figure(4);
surf(c./(f + fo), 1:N_trip, spec_z);
shading interp; axis tight; colorbar;
ylabel('Round-trip index'); xlabel('Wavelength (nm)');
zlabel('Normalised spectrum (a.u.)');
title('Spectral evolution across round trips');
view(0, 90);

phase_out = unwrap(angle(uout));
chirp = -diff(phase_out)/(2*pi*dt);
Eout = abs(uout).^2;
[~, Imax] = max(Eout); %#ok<ASGLU>
[width, I_l, I_r] = fwhm(Eout); %#ok<NASGU>

figure(5); clf;
[ax, p1, p2] = plotyy(t, Eout, t(1:end-1), chirp, 'plot', 'plot');
set(p1, 'Color', 'b', 'LineWidth', 1.5);
set(p2, 'Color', 'r', 'LineWidth', 1.2, 'LineStyle', '--');
xlabel(sprintf('Time (ps)   FWHM: %.2f ps', width*dt));
ylabel(ax(1), '|u(z,t)|^2 (W)'); ylabel(ax(2), 'Chirp (THz)');
grid on; axis(ax(1), 'tight'); axis(ax(2), 'tight');

title('Output intensity and instantaneous frequency');

%% ------------ Assemble final round-trip diagnostics for plotting -------
% replicate traces to follow layout of original diagnostic plots
ut_fft = repmat(abs(fftshift(fft(ut))), 20, 1);
uout_fft = repmat(abs(fftshift(fft(uout))), 20, 1);

Plotdata.ufft = [Plot_smf_link.ufft; Plot_amf_main.ufft; Plot_smf_output.ufft; ...
                 ut_fft; Plot_smf_pre.ufft; uout_fft; Plot_cfbg.ufft];

spec = abs(Plotdata.ufft').^2;
specnorm = spec ./ (lambda'*ones(1, size(spec,2))).^2;
specnorm = specnorm / max(specnorm(:));

figure(6);
surf(c./(f + fo), 1:size(Plotdata.ufft,1), specnorm');
shading interp; axis tight; colorbar;
xlabel('Wavelength (nm)'); ylabel('Segment index');
zlabel('Normalised spectral power');
title('Spectral evolution within the final round trip');
view(0, 90);

ut_t = repmat(ut, 20, 1);
uout_t = repmat(uout, 20, 1);
Plotdata.u = [Plot_smf_link.u; Plot_amf_main.u; Plot_smf_output.u; ...
              ut_t; Plot_smf_pre.u; uout_t; Plot_cfbg.u];

figure(7);
surf(t, 1:size(Plotdata.u,1), abs(Plotdata.u).^2);
shading interp; axis tight; colorbar;
xlabel('Time (ps)'); ylabel('Segment index');
zlabel('|u(z,t)|^2 (W)');
title('Temporal evolution within the final round trip');
view(0, 90);

figure(8);
yyaxis left;
plot(1:N_trip, spec_bw_hist, 'LineWidth', 1.4);
ylabel('Spectral FWHM (nm)');
yyaxis right;
plot(1:N_trip, energy_hist, '--', 'LineWidth', 1.1);
ylabel('Pulse energy (pJ)');
xlabel('Round trip index');
title('Mode-locking observables');
grid on;
legend({'Spectral width', 'Pulse energy'}, 'Location', 'best');

figure(9);
plot(1:N_trip, rho_hist, 1:N_trip, rho_out_hist, 1:N_trip, gain_hist);
legend({'NALM \rho', 'Output \rho_{out}', 'Main gain (dB)'}, 'Location', 'best');
xlabel('Round trip index');
ylabel('Tuned value');
title('Adaptive parameter history');
grid on;
