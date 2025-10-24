function [hrw,fr] = Raman_response_w(t,mod)
%RAMAN_RESPONSE_W Raman response in the frequency domain.

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
