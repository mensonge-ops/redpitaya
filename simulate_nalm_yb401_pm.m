function simulate_nalm_yb401_pm
    % simulate_nalm_yb401_pm.m
    % -------------------------------------------------------------------------
    % Numerical NALM mode-locking simulation configured for a cavity that
    % entirely relies on Nufern (Coherent) Yb401-PM ytterbium-doped fibre.
    % The implementation is derived from the reference CQEM/IP solver example
    % and keeps the same numerical core while updating the fibre, gain and
    % filtering parameters to reflect a realistic Yb-fibre cavity operating at
    % ~1030 nm.
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
    % cavity round trips and stores the intermediate data in Plotdata.*
    %
    % -------------------------------------------------------------------------

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
    amf_main.gssdB = 32;                    % small signal gain (dB)

    %% --------------------------- Filter section ----------------------------
    filter.lamda_c = lamda_pulse;           % central wavelength (nm)
    filter.landa_bw = 4;                    % 4 nm passband
    filter.fc = c/filter.lamda_c;           % THz
    filter.f3dB = c/(filter.lamda_c)^2*filter.landa_bw;
    filter.n = 1;                           % Gaussian filter order

    %% ------------------------------ Couplers -------------------------------
    rho = 0.55;                             % NALM coupler splitting ratio
    rho_out = 0.3;                          % output coupler

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
    w = 2*pi*f;

    %% -------------------------- Propagation step ---------------------------
    dz = 5e-6;                              % starting step size (km)
    tol = 1e-4;                             % adaptive tolerance

    %% ----------------------- Initial conditions ----------------------------
    P_peak = 2*N2*abs(ybfibre.betaw(3))/ybfibre.gamma/tfwhm^2;
    u0 = sqrt(P_peak)*sech(t/tfwhm);
    randn('state', 0);                      % reproducible seed
    u0 = (1 + 2e-3*randn(1,nt)).*u0;        % add weak noise to seed self-start

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
    N_trip = 40;                            % number of cavity round trips
    h1 = waitbar(0, 'Running Yb401-PM NALM simulation...');

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
        [ur, ut] = coupler(ub, uf, rho);
        u = ut;

        % propagation to output coupler: smf_pre spool reused for clarity
        [u, ~, Plot_smf_pre] = IP_CQEM_FD(u, dt, dz, smf_pre, fo, tol, 1, 1);
        [u, uout] = coupler(u, 0, rho_out);

        % spectral filter
        u = filter_gauss(u, filter.f3dB, filter.fc, filter.n, fo, df);

        % diagnostics
        figure(1); clf;
        plot(t, abs(u0).^2, 'b.-', t, abs(uout).^2, 'r.-'); axis tight;
        grid on;
        xlabel('Time (ps)'); ylabel('|u(z,t)|^2 (W)');
        title(sprintf('Seed (blue) vs. output (red) after trip %d', ii));

        spec = fftshift(abs(fft(uout)).^2);
        specnorm = spec ./ lambda.^2;
        specnorm = specnorm / max(specnorm);

        figure(2); clf;
        plot(c./(f + fo), specnorm, 'r.-'); axis tight; grid on;
        xlabel('Wavelength (nm)'); ylabel('Normalised spectrum (a.u.)');
        title(sprintf('Output spectrum after trip %d', ii));

        spec_z = [spec_z; specnorm];
        u_z = [u_z; uout];
    end

    close(h1);
    tx = toc;

    %% ------------------------------ Diagnostics ----------------------------
    fprintf('\nSimulation wall time (s) = %6.2f\n', tx);

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
    [~, Imax] = max(Eout);
    [width, I_l, I_r] = fwhm(Eout);
    range = max(I_l,1):min(I_r,length(t));

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
    u_f_fft = repmat(abs(fftshift(fft(u))), 20, 1);

    Plotdata.ufft = [Plot_smf_link.ufft; Plot_amf_main.ufft; Plot_smf_output.ufft; 
                     ut_fft; Plot_smf_pre.ufft; uout_fft; u_f_fft];

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
    u_f_t = repmat(u, 20, 1);
    Plotdata.u = [Plot_smf_link.u; Plot_amf_main.u; Plot_smf_output.u; 
                  ut_t; Plot_smf_pre.u; uout_t; u_f_t];

    figure(7);
    surf(t, 1:size(Plotdata.u,1), abs(Plotdata.u).^2);
    shading interp; axis tight; colorbar;
    xlabel('Time (ps)'); ylabel('Segment index');
    zlabel('|u(z,t)|^2 (W)');
    title('Temporal evolution within the final round trip');
    view(0, 90);

end
%% -------------------------- Local functions ----------------------------
function [u1o,u2o] = coupler(u1i,u2i,rho)
    rho = max(0,min(1,rho));
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

function [width,I_l,I_r] = fwhm(x)
    [peak, ind_peak] = max(x);
    half_peak = peak/2;
    x_l = x(1:ind_peak);
    x_r = x(ind_peak:end);
    I_l_rel = find(fliplr(x_l) <= half_peak, 1, 'first');
    I_r_rel = find(x_r <= half_peak, 1, 'first');
    width = I_l_rel + I_r_rel;
    I_l = ind_peak - I_l_rel;
    I_r = ind_peak + I_r_rel - 1;
end

function gain = gain_saturated2(Pin,gssdB,PsatdBm)
    gss = 10^(gssdB/10);
    Psat = 10^((PsatdBm-30)/10);
    gain = gss/(1+Pin/Psat);
end

function [u1,nf,Plotdata] = IP_CQEM_FD(u0,dt,dz,mod,fo,tol,dplot,quiet)
    nt = length(u0);
    w = fftshift(2*pi*(-nt/2:nt/2-1)/(dt*nt));
    t_vec = (-nt/2:1:nt/2-1)*dt;
    [hrw,fr] = Raman_response_w(t_vec,mod);

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

    if ~quiet
        fprintf(1,'\nSegment length %.1f cm ...', mod.L*1e2);
    end

    while propagedlength < mod.L
        if (dz + propagedlength) > mod.L
            dz = mod.L - propagedlength;
        end

        if isfield(mod,'gssdB')
            Pin0 = sum(u1.*conj(u1))/nt;
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

        uaux = halfstep.*(uip + k1/6 + k2/3 + k3/3) + k4/6;
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
        Plotdata.u = abs(u_z);
    else
        Plotdata = struct('z',[],'ufft',[],'u',[]);
    end
end

function [LOP] = Linearoperator_w(alpha,betaw,w)
    LOP = -fftshift(alpha/2);
    if length(betaw) == length(w)
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
