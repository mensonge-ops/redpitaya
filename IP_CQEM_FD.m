function [u1,nf,Plotdata] = IP_CQEM_FD(u0,dt,dz,mod,fo,tol,dplot,quiet)
%IP_CQEM_FD Interaction-picture CQEM solver for the GNLSE.

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
