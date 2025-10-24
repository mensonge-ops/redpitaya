function yb401pm_nalm_simulation()
%YB401PM_NALM_SIMULATION Simulate a NALM mode-locked fiber laser.
%   This script adapts the interaction-picture CQEM solver that is commonly
%   used for erbium-doped systems to the ytterbium-doped Yb401-PM fiber.  It
%   configures the passive and active segments, integrates the pulse around
%   the cavity for multiple roundtrips, and produces diagnostic plots for the
%   temporal and spectral evolution of the pulse.
%
%   The parameter set is based on typical manufacturer data for Yb401-PM:
%     * Center wavelength ~1030 nm with 30 nm emission bandwidth
%     * Mode-field diameter ~24 µm (Aeff ≈ 452 µm^2)
%     * Dispersion β2 ≈ 22 ps^2/km at 1030 nm (normal dispersion)
%     * Nonlinear coefficient derived from n2 = 2.6×10^-20 m^2/W
%     * Small-signal gain of 4.5 dB/m and saturation power ≈ 20 W
%
%   Running the simulation:
%     >> yb401pm_nalm_simulation
%
%   The script requires the helper functions bundled at the bottom of this
%   file.  They are direct MATLAB ports of the routines provided with the
%   original Er-doped example, rewritten here to keep the repository
%   self-contained.
%
%   Author: OpenAI ChatGPT

%% Fundamental constants and input pulse
c = 299792.458;                 % speed of light (nm/ps)
lamda_pulse = 1030;             % central wavelength for Yb lasers (nm)
fo = c/lamda_pulse;             % central frequency (THz)

N2 = 1^2;                       % soliton order
pulse.tfwhm = 2.5;              % FWHM duration (ps)

%% Build fiber modules using Yb401-PM parameters
params = yb401pm_params(lamda_pulse);

% Passive PM fiber used before the NALM coupler
pmf_pre = params.passive;
pmf_pre.L = 0.0018;               % 1.8 m

% Main gain fiber in the NALM loop (co-propagating path)
ydf_gain = params.active;
ydf_gain.L = 0.0012;            % 1.2 m active segment (≈ 4.5 dB/m)
ydf_gain.gssdB = ydf_gain.gssdB * (ydf_gain.L*1000); % scale per km

% Passive PM fiber after the gain segment to close the 4 m loop
pmf_post = params.passive;
pmf_post.L = 0.0015;              % 1.5 m

% Additional passive fiber section shared with the counter-propagating path
pmf_shared = params.passive;
pmf_shared.L = 0.002;             % 2.0 m

% NALM branch passive fiber
pmf_nalm = params.passive;
pmf_nalm.L = 0.0012;               % 1.2 m

%% Filter and coupler configuration
filter.lamda_c = lamda_pulse;
filter.landa_bw = 25;           % nm
filter.fc = c/filter.lamda_c;   % THz
filter.f3dB = c/(filter.lamda_c)^2 * filter.landa_bw;
filter.n = 1;

rho_loop = 0.4;                 % NALM coupler split ratio
rho_out = 0.2;                  % output coupler ratio

%% Numerical grid
nt = 2^12;                      % spectral points
window = 30;                    % ps time window

[t, f, lambda, w, dt, df] = build_time_frequency_grid(nt, window, lamda_pulse, c);

%% Initial pulse (sech) with quantum noise
P_peak = 2 * N2 * abs(pmf_pre.betaw(3)) / pmf_pre.gamma / pulse.tfwhm^2;
u = sqrt(P_peak) * sech(t/pulse.tfwhm);

randn('state', 0);
u = wgn(nt,1,25)'.*u;
PeakPower = max(abs(u).^2);

fprintf('\n----------------------------------------------\n');
fprintf('Initial peak power (W)  = %5.2f\n', PeakPower);
fprintf('Initial pulse energy (pJ) = %5.2f\n', dt * sum(abs(u).^2));

%% Cavity simulation
N_trip = 40;
dz = 1e-5;                     % km

spec_z = [];
u_z = [];

h_wait = waitbar(0, 'Simulating NALM roundtrips...');

for trip = 1:N_trip
    waitbar((trip-1)/N_trip, h_wait);

    % Shared passive segment prior to the NALM
    [u, ~] = IP_CQEM_FD(u, dt, dz, pmf_shared, fo, params.tol, 0, 0);

    % Split into counter-propagating fields at the loop coupler
    [uf, ub] = coupler(u, 0, rho_loop);

    % Forward path: passive -> gain -> passive
    [uf, ~] = IP_CQEM_FD(uf, dt, dz, pmf_pre, fo, params.tol, 0, 0);
    [uf, ~] = IP_CQEM_FD(uf, dt, dz, ydf_gain, fo, params.tol, 0, 0);
    [uf, ~] = IP_CQEM_FD(uf, dt, dz, pmf_post, fo, params.tol, 0, 0);

    % Backward path mirrors the forward path
    [ub, ~] = IP_CQEM_FD(ub, dt, dz, pmf_post, fo, params.tol, 0, 0);
    [ub, ~] = IP_CQEM_FD(ub, dt, dz, ydf_gain, fo, params.tol, 0, 0);
    [ub, ~] = IP_CQEM_FD(ub, dt, dz, pmf_pre, fo, params.tol, 0, 0);

    % Recombine at the loop coupler
    [u_loop, ~] = coupler(ub, uf, rho_loop);
    u = u_loop;

    % Propagate through the NALM output fiber
    [u, ~] = IP_CQEM_FD(u, dt, dz, pmf_nalm, fo, params.tol, 0, 0);

    % Output coupler: transmit part of the pulse out of the cavity
    [u, uout] = coupler(u, 0, rho_out);

    % Gaussian spectral filter
    u = filter_gauss(u, filter.f3dB, filter.fc, filter.n, fo, df);

    spec = fftshift(abs(fft(uout)).^2);
    specnorm = spec ./ lambda.^2;
    specnorm = specnorm / max(specnorm);

    spec_z = [spec_z; specnorm];
    u_z = [u_z; uout];

      if mod(trip, 5) == 0
          plot_evolution(t, u, uout, lambda);
      end
  end

close(h_wait);

%% Final diagnostics
  visualize_roundtrip(t, lambda, N_trip, u_z, spec_z);
plot_chirp(t, uout, dt);

fprintf('\nSimulation complete.\n');
end

%% Parameter helpers and diagnostics -------------------------------------------------
function params = yb401pm_params(lamda_pulse)
% Build passive and active module templates for Yb401-PM fiber.
params = struct();

params.n2 = 2.6;                    % nonlinear index in 10^-16 cm^2/W
params.Aeff = 452;                  % effective area in µm^2 (≈ 24 µm MFD)
params.gamma = 2*pi*params.n2/lamda_pulse/params.Aeff*1e4; % W^-1 km^-1
params.alpha = 3e-4;                % linear loss coefficient (km^-1)
params.betaw = [0 0 22 0.06e-3];    % dispersion coefficients (ps^n/nm)

passive = struct('Aeff', params.Aeff, ...
                 'n2', params.n2, ...
                 'gamma', params.gamma, ...
                 'alpha', params.alpha, ...
                 'L', 0.001, ...
                 'betaw', params.betaw, ...
                 'raman', 0, ...
                 'ssp', 0);

active = passive;
active.gssdB = 4.5;                 % dB per meter small-signal gain
active.PsatdBm = 43;                % ≈ 20 W saturation power
active.lamda_gain = lamda_pulse;
active.landa_bw = 30;
active.fc = 299792.458/active.lamda_gain;
active.fbw = 299792.458/(active.lamda_gain)^2 * active.landa_bw;

params.passive = passive;
params.active = active;
params.tol = 2e-4;
end

function [t, f, lambda, w, dt, df] = build_time_frequency_grid(nt, window, lamda_pulse, c)
dt = window/nt;
t = -window/2:dt:(window/2-dt);
df = 1/(nt*dt);
f = -(nt/2)*df:df:(nt/2-1)*df;
lambda = c./(f + c/lamda_pulse);
w = 2*pi*f;
end

function plot_evolution(t, u, uout, lambda)
figure(1); clf;
plot(t, abs(u).^2, 'b.-', t, abs(uout).^2, 'r.-');
xlabel('Time (ps)'); ylabel('|u(z,t)|^2 (W)');
title('Intracavity (blue) vs output (red) pulse');
grid on;

spec = fftshift(abs(fft(uout)).^2);
specnorm = spec ./ lambda.^2;
specnorm = specnorm/max(specnorm);
figure(2); clf;
plot(lambda, specnorm, 'r.-');
xlabel('Wavelength (nm)'); ylabel('Normalised spectrum');
title('Output spectrum');
grid on;
end

function visualize_roundtrip(t, lambda, N_trip, u_z, spec_z)
figure(3); clf;
surf(t, 1:N_trip, abs(u_z).^2);
shading interp; axis tight;
view(0, 90);
xlabel('Time (ps)'); ylabel('Roundtrip'); zlabel('|u(z,t)|^2 (W)');
title('Output temporal evolution');
colorbar;

figure(4); clf;
surf(lambda, 1:N_trip, spec_z);
shading interp; axis tight;
view(0, 90);
xlabel('Wavelength (nm)'); ylabel('Roundtrip'); zlabel('Normalised spectrum');
title('Output spectral evolution');
colorbar;
end

function plot_chirp(t, uout, dt)
phase_out = unwrap(angle(uout));
delta_w = -diff(phase_out)/(2*pi*dt);
Eout = abs(uout).^2;
[width, ~, ~] = fwhm(Eout);

figure(5); clf;
[ax, p1, p2] = plotyy(t, Eout, t(1:end-1), delta_w, 'plot', 'plot');
set(p1, 'Color', 'b', 'LineWidth', 1.5);
set(p2, 'Color', 'r', 'LineWidth', 1.2);
xlabel(ax(1), sprintf('Time (ps), FWHM %.2f ps', width*dt));
ylabel(ax(1), '|u(z,t)|^2 (W)');
ylabel(ax(2), 'Chirp (THz)');
legend({'Intensity', 'Chirp'}, 'Location', 'best');
axis(ax(1), 'tight'); axis(ax(2), 'tight');
end

%% Core numerical routines (ported helpers) ----------------------------------------
function [u1, nf, Plotdata] = IP_CQEM_FD(u0, dt, dz, mod, fo, tol, dplot, quiet)
nt = length(u0);
w = fftshift(2*pi*(-nt/2:nt/2-1)/(dt*nt));
t_vec = (-nt/2:nt/2-1)*dt; %#ok<NASGU>

[hrw, fr] = Raman_response_w(t_vec, mod);

ufft = fft(u0);
propagedlength = 0;
u1 = u0;
nf = 1;

if isfield(mod, 'gssdB')
    gain_w = filter_lorentz_tf(u1, mod.fbw, mod.fc, fo, 1/dt/nt);
    alpha_0 = mod.alpha;
end

if dplot == 1
    z_all = [];
    ufft_z = [];
    u_z = [];
end

while propagedlength < mod.L
    if (dz + propagedlength) > mod.L
        dz = mod.L - propagedlength;
    end

    if isfield(mod, 'gssdB')
        Pin0 = sum(u1.*conj(u1))/nt;
        gain = gain_saturated2(Pin0, mod.gssdB, mod.PsatdBm) .* gain_w;
        mod.alpha = alpha_0 - gain;
    end

    LOP = Linearoperator_w(mod.alpha, mod.betaw, w);

    PhotonN = sum((abs(ufft).^2)./(w + 2*pi*fo));
    PhotonN_z = sum(exp(-dz*fftshift(mod.alpha)).*(abs(ufft).^2)./(w + 2*pi*fo));

    halfstep = exp(LOP*dz/2);
    uip = halfstep.*ufft;
    k1 = halfstep*dz.*NonLinearoperator_w(u1, mod.gamma, w, fo, fr, hrw, dt, mod);

    uhalf2 = ifft(uip + k1/2);
    k2 = dz*NonLinearoperator_w(uhalf2, mod.gamma, w, fo, fr, hrw, dt, mod);

    uhalf3 = ifft(uip + k2/2);
    k3 = dz*NonLinearoperator_w(uhalf3, mod.gamma, w, fo, fr, hrw, dt, mod);

    uhalf4 = ifft(halfstep.*(uip + k3));
    k4 = dz*NonLinearoperator_w(uhalf4, mod.gamma, w, fo, fr, hrw, dt, mod);

    uaux = halfstep.*(uip + k1./6 + k2./3 + k3./3) + k4./6;

    propagedlength = propagedlength + dz;

    error = abs(sum((abs(uaux).^2)./(w+2*pi*fo)) - PhotonN_z)/PhotonN_z;
    if error > 2*tol
        propagedlength = propagedlength - dz;
        dz = dz/2;
    else
        ufft = uaux;
        u1 = ifft(ufft);
        if error > tol
            dz = dz/(2^0.2);
        elseif error < 0.5*tol
            dz = dz*(2^0.2);
        end

        if dplot == 1
            z_all = [z_all; propagedlength]; %#ok<AGROW>
            ufft_z = [ufft_z; abs(fftshift(ufft))]; %#ok<AGROW>
            u_z = [u_z; u1]; %#ok<AGROW>
        end
    end
    nf = nf + 16;
end

if dplot == 1
    Plotdata.z = z_all;
    Plotdata.ufft = ufft_z;
    Plotdata.u = u_z;
else
    Plotdata = struct('z', [], 'ufft', [], 'u', []);
end
end

function [u1o, u2o] = coupler(u1i, u2i, rho)
if rho > 1
    rho = 1;
elseif rho < 0
    rho = 0;
end

u1o = sqrt(rho)*u1i + 1i*sqrt(1-rho)*u2i;
u2o = 1i*sqrt(1-rho)*u1i + sqrt(rho)*u2i;
end

function uo = filter_gauss(ui, f3dB, fc, n, fo, df)
Ui = fft(ui);
N = numel(Ui);
f = fftshift((-(N/2)*df:df:(N/2-1)*df) + fo);
Tf = exp(-log(sqrt(2))*(2/f3dB*(f-fc)).^(2*n));
uo = ifft(Ui.*Tf);
end

function tf = filter_lorentz_tf(ui, fbw, fc, fo, df)
N = numel(ui);
f = (-(N/2)*df:df:(N/2-1)*df) + fo;
tf = (fbw)/2/pi./((f-fc).^2+(fbw/2)^2);
tf = tf/max(tf(:));
end

function [width, I_l, I_r] = fwhm(x)
[peak, ind_peak] = max(x);
half_peak = peak/2;
x_l = x(1:ind_peak);
x_r = x(ind_peak:end);
I_l = ind_peak - find(fliplr(x_l) <= half_peak, 1, 'first');
I_r = ind_peak + find(x_r <= half_peak, 1, 'first') - 1;
width = I_r - I_l;
end

function gain = gain_saturated2(Pin, gssdB, PsatdBm)
gss = 10^(gssdB/10);
Psat = (10^(PsatdBm/10))/1000;
gain = gss/(1+Pin/Psat);
end

function LOP = Linearoperator_w(alpha, betaw, w)
LOP = -fftshift(alpha/2);
if numel(betaw) == numel(w)
    LOP = LOP - 1i*betaw;
    LOP = fftshift(LOP);
else
    for ii = 0:numel(betaw)-1
        LOP = LOP - 1i*betaw(ii+1)*w.^ii/factorial(ii);
    end
end
end

function NLOP = NonLinearoperator_w(u_t, gamma, w, fo, fr, hrw, dt, mod)
if ~isfield(mod, 'ssp') || mod.ssp == 1
    NLOP = -1i*gamma*(1 + w/(2*pi*fo)).*fft(((1-fr)*u_t.*abs(u_t).^2) + ...
        fr*dt*u_t.*ifft(hrw.*fft(abs(u_t).^2)));
else
    NLOP = -1i*gamma*fft(((1-fr)*u_t.*abs(u_t).^2) + ...
        fr*dt*u_t.*ifft(hrw.*fft(abs(u_t).^2)));
end
end

function [hrw, fr] = Raman_response_w(t, mod)
if isfield(mod, 'raman') && mod.raman == 0
    hrw = 0;
    fr = 0;
else
    t1 = 12.2e-3;
    t2 = 32e-3;
    tb = 96e-3;
    fc = 0.04; %#ok<NASGU>
    fb = 0.21;
    fa = 1 - fc - fb;
    fr = 0.245;

    tres = t - t(1);

    ha = ((t1^2 + t2^2)/(t1*t2^2)).*exp(-tres/t2).*sin(tres/t1);
    hb = ((2*tb - tres)./tb^2).*exp(-tres/tb);
    hr = (fa + fc)*ha + fb*hb;

    hrw = fft(hr);
end
end
