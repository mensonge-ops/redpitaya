% yb401pm_nalm_sim.m
% Simulation of a NALM mode-locked fiber laser based on Yb401-PM fiber
% The model follows the interaction-picture CQEM solver that is commonly
% used for ultrafast fiber laser design.  The implementation adapts the
% example provided for Er-doped systems to match the dispersion, nonlinear
% and gain characteristics of the Yb401-PM fiber platform.

clear

%% ***************************** 输入参数 *********************************
% Fundamental constants
c = 299792.458;                         % speed of light (nm/ps)

% Seed pulse configuration
N2 = 1.2^2;                              % soliton order (dimensionless)
tfwhm = 600;                             % pulse FWHM (fs)
tfwhm = tfwhm * 1e-3;                    % convert to ps
lambda_pulse = 1030;                     % central wavelength (nm)
fo = c/lambda_pulse;                     % central frequency (THz)

%% ***************************** 光纤参数 *********************************
% Yb401-PM passive fiber approximations
Aeff_passive = 32.7;                     % effective mode area (um^2)
n2_silica = 2.6;                         % nonlinear coefficient (10^-16 cm^2/W)
gamma_passive = 2*pi*n2_silica/lambda_pulse/Aeff_passive*1e4; % (W^-1 km^-1)
alpha_passive = 0.3;                     % linear loss (dB/km)
alpha_passive = alpha_passive/log10(exp(1)); % convert to 1/km

% Dispersion for HI-1060 like passive fiber (ps^n/nm)
D_1060 = 18;                             % ps/(nm km) @ 1030 nm
beta2 = -(lambda_pulse^2/(2*pi*c))*D_1060; % ps^2/km
beta3 = 0.08;                             % ps^3/km (approx.)

base_fiber.betaw = [0 0 beta2 beta3*1e-3]; % include beta2 and beta3
base_fiber.gamma = gamma_passive;
base_fiber.alpha = alpha_passive;
base_fiber.Aeff = Aeff_passive;
base_fiber.n2 = n2_silica;
base_fiber.raman = 0;                    % disable Raman for simplicity
base_fiber.ssp = 0;                      % disable self-steepening

% SMF segments inside NALM loop
smf_loop = base_fiber;
smf_loop.L = 0.0012;                      % 1.2 m passive segment

% Passive delivery fiber after loop
smf_output = base_fiber;
smf_output.L = 0.0008;                    % 0.8 m

% Passive fiber before NALM
smf_pre = base_fiber;
smf_pre.L = 0.001;                        % 1 m

% Passive pigtail after filter
smf_pigtail = base_fiber;
smf_pigtail.L = 0.0005;                   % 0.5 m

%% ***************************** 增益段 ***********************************
% Yb401-PM gain fiber parameters (1.2 m length typical)
Aeff_gain = 32.7;                        % assume same mode area
amplifier = base_fiber;
amplifier.Aeff = Aeff_gain;
amplifier.gamma = 2*pi*n2_silica/lambda_pulse/Aeff_gain*1e4;
amplifier.L = 0.0012;                     % 1.2 m active fiber in loop
amplifier.gssdB = 25;                    % small signal gain (dB)
amplifier.PsatdBm = 36;                  % saturation power (dBm)
amplifier.lambda_gain = lambda_pulse;
amplifier.lambda_bw = 40;                % FWHM of gain bandwidth (nm)
amplifier.fc = c/amplifier.lambda_gain;
amplifier.fbw = c/(amplifier.lambda_gain)^2*amplifier.lambda_bw;

% Output-side power amplifier (optional)
amplifier2 = amplifier;
amplifier2.L = 0.0008;                     % 0.8 m gain fiber
amplifier2.gssdB = 18;                    % slightly lower gain

%% ***************************** 滤波器 ***********************************
filter.lamda_c = lambda_pulse;
filter.landa_bw = 12;                     % spectral filter bandwidth (nm)
filter.fc = c/filter.lamda_c;
filter.f3dB = c/(filter.lamda_c)^2*filter.landa_bw;
filter.n = 1;

%% ***************************** 耦合器参数 *******************************
rho_nalm = 0.4;                           % NALM coupler splitting ratio
rho_out = 0.2;                            % output coupler ratio

%% ***************************** 数值计算参数 *****************************
nt = 2^12;                                % number of temporal samples
time_window = 80;                         % ps

dt = time_window/nt;
t = -time_window/2:dt:time_window/2-dt;

df = 1/(nt*dt);
f = -(nt/2)*df:df:(nt/2-1)*df;
lambda = c./(f + fo);
w = 2*pi*f;

dz = 5e-6;                                % km

%% ***************************** 初始脉冲 *********************************
P_peak = 2*N2*abs(base_fiber.betaw(3))/base_fiber.gamma/tfwhm^2;
u0 = sqrt(P_peak)*sech(t/tfwhm);
randn('state', 1);
u0 = wgn(nt,1,20)'.*u0;                   % add noise
PeakPower = max(abs(u0).^2);

fprintf('\n----------------------------------------------\n');
fprintf('输入峰值功率 (W) = %5.2f\n', PeakPower);
fprintf('输入脉冲能量 (pJ) = %5.2f\n', dt*sum(abs(u0).^2));

%% ***************************** 模拟循环 ********************************
N_trip = 30;

tol = 2e-4;
u = u0;

spec_z = [];
u_z = [];

fprintf('\n开始模拟基于Yb401-PM的NALM锁模激光器...\n');

tstart = tic;

for roundtrip = 1:N_trip
    % Pre-loop passive fiber
    [u, ~, Plot_smf_pre] = IP_CQEM_FD(u, dt, dz, smf_pre, fo, tol, 1, 0);

    % Loop coupler split
    [uf, ub] = coupler(u, 0, rho_nalm);

    % Forward path: passive -> gain -> passive
    [uf, ~, ~] = IP_CQEM_FD(uf, dt, dz, smf_loop, fo, tol, 1, 0);
    [uf, ~, ~] = IP_CQEM_FD(uf, dt, dz, amplifier, fo, tol, 1, 0);
    [uf, ~, ~] = IP_CQEM_FD(uf, dt, dz, smf_loop, fo, tol, 1, 0);

    % Backward path (reverse order)
    [ub, ~, ~] = IP_CQEM_FD(ub, dt, dz, smf_loop, fo, tol, 1, 0);
    [ub, ~, ~] = IP_CQEM_FD(ub, dt, dz, amplifier, fo, tol, 1, 0);
    [ub, ~, ~] = IP_CQEM_FD(ub, dt, dz, smf_loop, fo, tol, 1, 0);

    % Recombine at loop coupler
    [u_loop, u_ref] = coupler(ub, uf, rho_nalm);
    u = u_loop;

    % Post-loop gain section (power amplifier)
    [u, ~, Plot_amp2] = IP_CQEM_FD(u, dt, dz, amplifier2, fo, tol, 1, 0);

    % Output segment and coupler
    [u, ~, Plot_smf_out] = IP_CQEM_FD(u, dt, dz, smf_output, fo, tol, 1, 1);
    [u, uout] = coupler(u, 0, rho_out);

    % Gaussian bandpass filter
    u = filter_gauss(u, filter.f3dB, filter.fc, filter.n, fo, df);

    % Passive pigtail
    [u, ~, Plot_smf_tail] = IP_CQEM_FD(u, dt, dz, smf_pigtail, fo, tol, 1, 1);

    % Store evolution
    spec = fftshift(abs(fft(uout)).^2);
    spec_norm = spec./lambda.^2;
    spec_norm = spec_norm / max(spec_norm);
    spec_z = [spec_z; spec_norm];
    u_z = [u_z; uout];

    if mod(roundtrip,5)==0
        fprintf('Roundtrip %d 完成\n', roundtrip);
    end
end

tsim = toc(tstart);
fprintf('\n模拟耗时 (秒) = %5.2f\n', tsim);

%% ***************************** 可视化 ***********************************
figure(1);
plot(t, abs(u0).^2, 'b', t, abs(uout).^2, 'r');
xlabel('时间 (ps)'); ylabel('|u(t)|^2 (W)');
title('初始脉冲与输出脉冲比较'); grid on;
legend('输入', '输出');

figure(2);
surf(c./(f+fo), 1:N_trip, spec_z, 'EdgeColor', 'none');
xlabel('波长 (nm)'); ylabel('循环次数'); zlabel('归一化光谱');
title('输出光谱演化'); view(0,90); colorbar;

figure(3);
surf(t, 1:N_trip, abs(u_z).^2, 'EdgeColor', 'none');
xlabel('时间 (ps)'); ylabel('循环次数'); zlabel('|u|^2 (W)');
title('输出脉冲演化'); view(0,90); colorbar;

%% ***************************** 函数定义 ********************************
function [u1o,u2o] = coupler(u1i,u2i,rho)
    rho = max(0,min(1,rho));
    u1o = sqrt(rho)*u1i + 1i*sqrt(1-rho)*u2i;
    u2o = 1i*sqrt(1-rho)*u1i + sqrt(rho)*u2i;
end

function uo = filter_gauss(ui,f3dB,fc,n,fo,df)
    Ui = fft(ui);
    N = numel(Ui);
    f = fftshift((-(N/2)*df:df:(N/2-1)*df) + fo);
    Tf = exp(-log(sqrt(2))*(2/f3dB*(f-fc)).^(2*n));
    uo = ifft(Ui.*Tf);
end

function tf = filter_lorentz_tf(ui,fbw,fc,fo,df)
    N = numel(ui);
    f = (-(N/2)*df:df:(N/2-1)*df) + fo;
    tf = (fbw)/2/pi./((f-fc).^2+(fbw/2)^2);
    tf = tf/max(tf(:));
end

function gain = gain_saturated2(Pin,gssdB,PsatdBm)
    gss = 10^(gssdB/10);
    Psat = (10^(PsatdBm/10))/1000;
    gain = gss/(1+Pin/Psat);
end

function [u1,nf,Plotdata] = IP_CQEM_FD(u0,dt,dz,mod,fo,tol,dplot,quiet)
    nt = numel(u0);
    w = fftshift(2*pi*(-nt/2:nt/2-1)/(dt*nt));
    t = (-nt/2:nt/2-1)*dt;

    [hrw,fr] = Raman_response_w(t,mod);

    ufft = fft(u0);
    propagedlength = 0;
    u1 = u0;
    nf = 1;

    if isfield(mod,'gssdB')
        gain_w = filter_lorentz_tf(u1,mod.fbw,mod.fc,fo,1/dt/nt);
        alpha_0 = mod.alpha;
    end

    if dplot == 1
        z_all = [];
        ufft_z = [];
        u_z = [];
    end

    while propagedlength < mod.L
        if dz + propagedlength > mod.L
            dz = mod.L - propagedlength;
        end

        if isfield(mod,'gssdB')
            Pin0 = (sum(u1.*conj(u1))/nt);
            gain = gain_saturated2(Pin0,mod.gssdB,mod.PsatdBm).*gain_w;
            mod.alpha = alpha_0 - gain;
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
                z_all = [z_all; propagedlength];
                ufft_z = [ufft_z; abs(fftshift(ufft))];
                u_z = [u_z; u1];
            end
        end
        nf = nf + 16;
    end

    if dplot == 1
        Plotdata.z = z_all;
        Plotdata.ufft = ufft_z;
        Plotdata.u = abs(u_z);
    else
        Plotdata = 0;
    end
end

function [LOP] = Linearoperator_w(alpha,betaw,w)
    LOP = -fftshift(alpha/2);
    if numel(betaw) == numel(w)
        LOP = LOP - 1i*betaw;
        LOP = fftshift(LOP);
    else
        for ii = 0:numel(betaw)-1
            LOP = LOP - 1i*betaw(ii+1)*(w).^ii/factorial(ii);
        end
    end
end

function [NLOP] = NonLinearoperator_w(u_t,gamma,w,fo,fr,hrw,dt,mod)
    if ~isfield(mod,'ssp') || mod.ssp == 1
        NLOP = -1i*gamma*(1 + w/(2*pi*fo)).*fft(((1-fr)*u_t.*abs(u_t).^2) ...
            + fr*dt*u_t.*ifft(hrw.*fft(abs(u_t).^2)));
    else
        NLOP = -1i*gamma*fft(((1-fr)*u_t.*abs(u_t).^2) ...
            + fr*dt*u_t.*ifft(hrw.*fft(abs(u_t).^2)));
    end
end

function [hrw,fr] = Raman_response_w(t,mod)
    if isfield(mod,'raman') && mod.raman == 0
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
        tres = t - t(1);
        ha = ((t1^2+t2^2)/(t1*t2^2)).*exp(-tres/t2).*sin(tres/t1);
        hb = ((2*tb - tres)./tb^2).*exp(-tres/tb);
        hr = (fa + fc)*ha + fb*hb;
        hrw = fft(hr);
    end
end

