% NALM mode-locked fiber laser simulation based on Yb401-PM fiber parameters
% This script is adapted from a generalized CQEM-based cavity solver.
% It configures the loop to emulate a Yb-doped NALM laser operating near 1030 nm
% using typical specification data for nLIGHT Liekki Yb401-PM polarization-
% maintaining fiber.
%
% The script keeps the numerical core identical to the reference implementation
% while updating gain, dispersion and cavity layout to match the Yb system.  All
% helper functions required for the simulation are included at the end of the
% file so the script can be executed directly in MATLAB or Octave.

clear

%% ***************************** Input parameters *****************************
% Physical constants
c = 299792.458;                             % Speed of light (nm/ps)

% Input pulse
N2 = 0.8^2;                                 % Soliton order (dimensionless)
tfwhm = 20;                                 % Pulse full-width at half maximum (ps)
lamda_pulse = 1030;                         % Pulse central wavelength (nm)
fo = c/lamda_pulse;                         % Central frequency (THz)

%% Yb401-PM passive fiber parameters (values derived from datasheet averages)
yb.Aeff = 78.5;                             % Effective area (um^2) ~ pi*(5 um)^2
yb.n2 = 2.6;                                % Nonlinear Kerr coeff. (1e-16 cm^2/W)
yb.gamma = 2*pi*yb.n2/lamda_pulse/yb.Aeff*1e4;  % Nonlinear coeff. (W^-1 km^-1)
yb.alpha = 0.2/4.343;                       % Attenuation 0.2 dB/km -> km^-1
yb.betaw = [0 0 22.1 0.12e-3];              % Dispersion poly (ps^n/nm)
yb.raman = 0;                               % Disable Raman contribution
yb.ssp = 0;                                 % Disable self-steepening

%% Passive fiber sections in the cavity
smf_nalm = yb;                              % Fiber used inside NALM loop
smf_nalm.L = 0.0006;                        % 0.6 m section before gain in loop

smf_tail = yb;                              % Output delivery section
smf_tail.L = 0.0012;                        % 1.2 m passive fiber after NALM

smf_splice = yb;                            % Short splice between gain stages
smf_splice.L = 0.00025;                     % 0.25 m

%% Yb401-PM gain fiber parameters (two segments forming the NALM)
gain = yb;
gain.L = 0.00035;                           % 0.35 m active fiber length
gain.gssdB = 23;                            % Small-signal gain (dB)
gain.PsatdBm = 32;                           % Saturation power (~1.6 W)
gain.lamda_gain = lamda_pulse;              % Gain peak (nm)
gain.landa_bw = 40;                         % Gain bandwidth FWHM (nm)
gain.fc = c/gain.lamda_gain;                % Gain center frequency (THz)
gain.fbw = c/(gain.lamda_gain)^2*gain.landa_bw;  % Gain bandwidth (THz)

post_gain = gain;                           % Booster gain fiber after the loop
post_gain.L = 0.00045;                      % 0.45 m
post_gain.gssdB = 28;                       % Slightly larger gain for output

%% Spectral filtering
filter.lamda_c = lamda_pulse;
filter.landa_bw = 10;                       % 10 nm filter bandwidth
filter.fc = c/filter.lamda_c;
filter.f3dB = c/(filter.lamda_c)^2*filter.landa_bw;
filter.n = 1;                               % First-order Gaussian filter

%% Coupler ratios
rho_nalm = 0.5;                             % NALM coupler splitting ratio
rho_out = 0.3;                              % Output coupler ratio

%% Numerical grid
nt = 2^11;                                  % Number of temporal samples
time = 40;                                  % Simulation window (ps)
dt = time/nt;                               % Time resolution (ps)
t = -time/2:dt:(time/2-dt);                 % Time vector (ps)

df = 1/(nt*dt);                             % Frequency spacing (THz)
f = -(nt/2)*df:df:(nt/2-1)*df;              % Frequency vector (THz)
lambda = c./(f + fo);                       % Wavelength vector (nm)
w = 2*pi*f;                                 % Angular frequency (rad/ps)

dz = 5e-6;                                  % Initial propagation step (km)
tol = 2e-4;                                 % Photon error tolerance

%% **************************** Initial field ******************************
beta2 = smf_nalm.betaw(3);
P_peak = 2*N2*abs(beta2)/smf_nalm.gamma/tfwhm^2;  % Analytical soliton peak power
u0 = sqrt(P_peak)*sech(t/tfwhm);                 % Initial hyperbolic secant field
randn('state', 0);
u0 = wgn(nt,1,25)'.*u0;                              % Additive noise seed
PeakPower = max(abs(u0).^2);

fprintf('\n----------------------------------------------\n');
fprintf('Input peak power (W) = %5.2f\n', PeakPower);
fprintf('Input pulse energy (pJ) = %5.2f\n', dt*sum(abs(u0).^2));

%% ************************* Propagation loop ******************************
fprintf('\nStarting CQEM simulation for Yb401-PM NALM...\n');
tic;

spec_z = [];
u_z = [];

u = u0;
N_trip = 30;                                % Number of round-trips to simulate

h_wait = waitbar(0,'Running Yb401-PM NALM simulation...');

for ii = 1:N_trip
    waitbar((ii-1)/N_trip, h_wait);

    % Passive section before the NALM
    [u, ~, data_smf_head] = IP_CQEM_FD(u, dt, dz, smf_nalm, fo, tol, 1, 0);

    % Forward/backward splitting at the NALM coupler
    [uf, ub] = coupler(u, 0, rho_nalm);

    % Forward path: gain -> splice fiber -> gain
    [u_forward, ~, data_gain_f] = IP_CQEM_FD(uf, dt, dz, gain, fo, tol, 1, 0);
    [u_forward, ~, data_splice_f] = IP_CQEM_FD(u_forward, dt, dz, smf_splice, fo, tol, 1, 0);
    [u_forward, ~, data_gain_f2] = IP_CQEM_FD(u_forward, dt, dz, gain, fo, tol, 1, 0);

    % Backward path (reverse order for counter-propagating arm)
    [u_backward, ~, data_gain_b] = IP_CQEM_FD(ub, dt, dz, gain, fo, tol, 1, 0);
    [u_backward, ~, data_splice_b] = IP_CQEM_FD(u_backward, dt, dz, smf_splice, fo, tol, 1, 0);
    [u_backward, ~, data_gain_b2] = IP_CQEM_FD(u_backward, dt, dz, gain, fo, tol, 1, 0);

    % Recombine at the coupler to obtain transmitted field
    [~, ut] = coupler(u_backward, u_forward, rho_nalm);
    u = ut;

    % Tail fiber, booster gain and delivery fiber
    [u, ~, data_smf_tail] = IP_CQEM_FD(u, dt, dz, smf_tail, fo, tol, 1, 0);
    [u, ~, data_booster] = IP_CQEM_FD(u, dt, dz, post_gain, fo, tol, 1, 0);

    % Output coupler and filtering
    [u, uout] = coupler(u, 0, rho_out);
    u = filter_gauss(u, filter.f3dB, filter.fc, filter.n, fo, df);

    % Diagnostics
    figure(1); clf;
    plot(t, abs(u0).^2, 'b.-', t, abs(uout).^2, 'r.-'); grid on; axis tight;
    xlabel('Time (ps)'); ylabel('|u(t)|^2 (W)');
    title(sprintf('Initial (blue) vs output (red) pulse after trip %d', ii));

    spec = fftshift(abs(fft(uout)).^2);
    specnorm = spec ./ lambda.^2;
    specnorm = specnorm / max(specnorm);
    figure(2); clf;
    plot(c./(f + fo), specnorm, 'r.-'); grid on; axis tight;
    xlabel('Wavelength (nm)'); ylabel('Normalized spectrum (a.u.)');
    title(sprintf('Output spectrum after trip %d', ii));

    spec_z = [spec_z; specnorm];
    u_z = [u_z; uout];
end

close(h_wait);
sim_time = toc;

ut_final = ut;                                   % Store final NALM transmission
u_filtered_final = u;                            % Store final intra-cavity field

fprintf('\nSimulation completed in %.2f s\n', sim_time);

%% ******************************* Visualisation *****************************
figure(3); clf;
surf(t, 1:N_trip, abs(u_z).^2, 'EdgeColor', 'none');
colorbar; axis tight; shading interp;
ylabel('Round-trip'); xlabel('Time (ps)'); zlabel('|u|^2 (W)');
title('Temporal evolution of the circulating pulse');
view(0,90);

figure(4); clf;
surf(c./(f + fo), 1:N_trip, spec_z, 'EdgeColor', 'none');
colorbar; axis tight; shading interp;
ylabel('Round-trip'); xlabel('Wavelength (nm)'); zlabel('Normalized spectrum');
title('Spectral evolution of the output field');
view(0,90);

%% Extract chirp characteristics for the final trip
phase_out = unwrap(angle(uout));
delta_w = diff(phase_out)/(dt*2*pi);            % Instantaneous frequency shift (THz)
Eout = abs(uout).^2;
[~, idx_max] = max(Eout);
[width, ~, ~] = fwhm(Eout);
range = floor((idx_max-1*width):(idx_max+0.9*width));
range = range(range >= 1 & range <= numel(t));
if numel(range) < 2
    range = 1:(numel(t)-1);
end
range_delta = range(range <= numel(delta_w));
if isempty(range_delta)
    range_delta = 1:numel(delta_w);
end

figure(5); clf;
[ax, p1, p2] = plotyy(t, Eout, t(range_delta), delta_w(range_delta), 'plot', 'plot');
set(p1, 'LineWidth', 1.5);
set(p2, 'LineWidth', 1.5, 'LineStyle', '--');
xlabel(sprintf('Time (ps), FWHM %.2f ps', width*dt));
ylabel(ax(1), '|u(t)|^2 (W)'); ylabel(ax(2), 'Chirp (THz)');
grid on; axis tight;

%% Assemble propagation data for distance-resolved plots (final trip)
data_segments = {
    data_smf_head;
    data_gain_f; data_splice_f; data_gain_f2;
    data_smf_tail; data_booster
    };

segment_physical_lengths = [
    smf_nalm.L,
    gain.L,
    smf_splice.L,
    gain.L,
    smf_tail.L,
    post_gain.L
    ];

Plotdata.ufft = [];
Plotdata.u = [];
segment_lengths = zeros(numel(data_segments),1);

for idx = 1:numel(data_segments)
    seg = data_segments{idx};
    if ~isstruct(seg)
        continue;
    end
    Plotdata.ufft = [Plotdata.ufft; seg.ufft];
    Plotdata.u = [Plotdata.u; seg.u];
    segment_lengths(idx) = size(seg.ufft,1);
end

% Append output coupler / filter diagnostics
ut_fft = repmat(abs(fftshift(fft(ut_final))), 10, 1);
u_filtered_fft = repmat(abs(fftshift(fft(u_filtered_final))), 10, 1);
Plotdata.ufft = [Plotdata.ufft; ut_fft; u_filtered_fft];
Plotdata.u = [Plotdata.u; repmat(ut_final, 10, 1); repmat(u_filtered_final, 10, 1)];
segment_lengths = [segment_lengths; 10; 10];
segment_physical_lengths = [segment_physical_lengths, 1e-9, 1e-9];

% Build cumulative distance vector
segments = segment_lengths(:)';
z_vector = zeros(1, sum(segments));

cursor = 0;
prop_distance = 0;
for ii = 1:numel(segment_lengths)
    seg_len = segment_lengths(ii);
    if seg_len == 0
        continue;
    end
    z_vector(cursor + (1:seg_len)) = linspace(prop_distance, ...
        prop_distance + segment_physical_lengths(ii), seg_len);
    prop_distance = prop_distance + segment_physical_lengths(ii);
    cursor = cursor + seg_len;
end

if cursor < numel(z_vector)
    delta = 1e-9;
    remaining = numel(z_vector) - cursor;
    z_vector(cursor+1:end) = prop_distance + (0:remaining-1)*delta;
end

figure(6); clf;
[F_mesh, Z_mesh] = meshgrid(c./(f + fo), z_vector);
spec_final = abs(Plotdata.ufft).^2;
spec_norm = spec_final ./ (lambda'.^2);
spec_norm = spec_norm / max(spec_norm(:));
surf(F_mesh, Z_mesh, spec_norm, 'EdgeColor', 'none');
view(-30,30); grid on;
xlabel('Wavelength (nm)'); ylabel('Propagation distance (km)');
zlabel('Normalized spectral power');
title('Intra-cavity spectral evolution (final trip)');
colorbar;

figure(7); clf;
[T_mesh, Z_mesh] = meshgrid(t, z_vector);
intensity = abs(Plotdata.u).^2;
surf(T_mesh, Z_mesh, intensity, 'EdgeColor', 'none');
view(-30,30); grid on;
xlabel('Time (ps)'); ylabel('Propagation distance (km)');
zlabel('Intensity (W)');
title('Intra-cavity temporal evolution (final trip)');
colorbar;

%% Helper functions #########################################################
function [u1,nf,Plotdata] =  IP_CQEM_FD(u0,dt,dz,mod,fo,tol,dplot,quiet)
    nt = length(u0);
    w = fftshift(2*pi*(-nt/2:nt/2-1)/(dt*nt));
    t_vec = (-nt/2:1:nt/2-1)*dt; %#ok<NASGU>
    [hrw,fr] = Raman_response_w(t_vec,mod);

    ufft = fft(u0);
    propagedlength = 0;
    u1 = u0;
    nf = 1;

    if isfield(mod,'gssdB')
        gain_w = filter_lorentz_tf(u1,mod.fbw,mod.fc,fo,1/dt/nt);
        alpha_0 = mod.alpha;
    end

    if dplot ==1
        z_all = [];
        ufft_z = [];
        u_z = [];
    end

    while propagedlength < mod.L
        if (dz + propagedlength) > mod.L
            dz = mod.L - propagedlength;
        end

        if isfield(mod,'gssdB')
            Pin0 = (sum(u1.*conj(u1))/nt);
            gain = gain_saturated2(Pin0,mod.gssdB,mod.PsatdBm).*gain_w;
            mod.alpha = alpha_0-gain;
        end
        LOP = Linearoperator_w(mod.alpha,mod.betaw,w);

        PhotonN = sum((abs(ufft).^2)./(w + 2*pi*fo));
        PhotonN_z = sum(exp(-dz*fftshift(mod.alpha)).*(abs(ufft).^2)./(w + 2*pi*fo));

        halfstep = exp(LOP*dz/2);
        uip = halfstep.*ufft;
        k1 = halfstep*dz.*NonLinearoperator_w(u1,mod.gamma,w,fo,fr,hrw,dt,mod);

        uhalf2 = ifft(uip + k1/2);
        k2 = dz*NonLinearoperator_w(uhalf2,mod.gamma,w,fo,fr,hrw,dt,mod);

        uhalf3 = ifft(uip + k2/2);
        k3 = dz*NonLinearoperator_w(uhalf3,mod.gamma,w,fo,fr,hrw,dt,mod);

        uhalf4 = ifft(halfstep.*(uip + k3));
        k4 = dz*NonLinearoperator_w(uhalf4,mod.gamma,w,fo,fr,hrw,dt,mod);

        uaux = halfstep.*(uip + k1./6 + k2./3 + k3./3) + k4./6;

        propagedlength = propagedlength + dz;

        error = abs(sum((abs(uaux).^2)./(w+2*pi*fo))-PhotonN_z)/PhotonN_z;
        if error > 2 * tol
            propagedlength = propagedlength - dz;
            dz = dz/2;
        else
            ufft = uaux;
            u1 = ifft(ufft);
            if error > tol
                dz = dz/(2^0.2);
            else
                if error < 0.5*tol
                    dz = dz*(2^0.2);
                end
            end
            if dplot ==1
                z_all = [z_all;propagedlength];
                ufft_z = [ufft_z;abs(fftshift(ufft))];
                u_z = [u_z;u1];
            end
        end
        nf = nf + 16;
    end

    if dplot ==1
        Plotdata.z = z_all;
        Plotdata.ufft = ufft_z;
        Plotdata.u = abs(u_z);
    else
        Plotdata = struct('z',[],'ufft',[],'u',[]);
    end
end

function [u1o,u2o] = coupler(u1i,u2i,rho)
    if rho>1
        rho = 1;
    elseif rho <0
        rho = 0;
    end

    u1o = sqrt(rho)*u1i + 1i*sqrt(1-rho)*u2i;
    u2o = 1i*sqrt(1-rho)*u1i + sqrt(rho)*u2i;
end

function uo = filter_gauss(ui,f3dB,fc,n,fo,df)
    Ui = fft(ui);
    N = size(Ui,2);
    f = fftshift((-(N/2)*df:df:(N/2-1)*df) + fo);
    Tf = exp(-log(sqrt(2))*(2/f3dB*(f-fc)).^(2*n));
    uo = ifft(Ui.*Tf);
end

function tf = filter_lorentz_tf(ui,fbw,fc,fo,df)
    N = size(ui,2);
    f = (-(N/2)*df:df:(N/2-1)*df) + fo;
    tf = (fbw)/2/pi./((f-fc).^2+(fbw/2)^2);
    tf = tf/max(tf(:));
end

function gain = gain_saturated2(Pin,gssdB,PsatdBm)
    gss = 10^(gssdB/10);
    Psat = (10^(PsatdBm/10))/1000;
    gain = gss/(1+Pin/Psat);
end

function [LOP] = Linearoperator_w(alpha,betaw,w)
    LOP = -fftshift(alpha/2);
    if (length(betaw) == length(w))
        LOP = LOP - 1i*betaw;
        LOP = fftshift(LOP);
    else
        for ii = 0:length(betaw)-1
            LOP = LOP - 1i*betaw(ii+1)*(w).^ii/factorial(ii);
        end
    end
end

function [NLOP] = NonLinearoperator_w(u_t,gamma,w,fo,fr,hrw,dt,mod)
    if ~isfield(mod,'ssp') || mod.ssp == 1
        NLOP = -1i*gamma*(1 + w/(2*pi*fo)).*fft(((1-fr)*u_t.*abs(u_t).^2)...
            + fr*dt*u_t.*ifft(hrw.*fft(abs(u_t).^2)));
    else
        NLOP = -1i*gamma*fft(((1-fr)*u_t.*abs(u_t).^2)...
            + fr*dt*u_t.*ifft(hrw.*fft(abs(u_t).^2)));
    end
end

function [hrw,fr] = Raman_response_w(t,mod)
    if isfield(mod,'raman') && mod.raman==0
        hrw = 0;
        fr = 0;
    else
        t1 = 12.2e-3;
        t2 = 32e-3;
        tb = 96e-3;
        fc = 0.04;
        fb = 0.21;
        fa = 1 - fc - fb;
        fr = 0.245;

        tres = t-t(1);
        ha =((t1^2+t2^2)/(t1*t2^2)).*exp(-tres/t2).*sin(tres/t1);
        hb = ((2*tb - tres)./tb^2).*exp(-tres/tb);
        hr = (fa + fc)*ha + fb*hb;
        hrw = fft(hr);
    end
end

function [width,I_l,I_r] = fwhm(x)
    [peak, ind_peak] = max(x);
    half_peak = peak/2;
    x_l = x(1:ind_peak);
    x_r = x(1+ind_peak:end);
    I_l = find(fliplr(x_l)<=half_peak,1,'first');
    I_r = find(x_r<=half_peak,1,'first');
    width = I_l+I_r;
    I_l = ind_peak - I_l;
    I_r = ind_peak + I_r;
end
