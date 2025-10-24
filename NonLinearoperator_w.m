function NLOP = NonLinearoperator_w(u_t,gamma,w,fo,fr,hrw,dt,mod)
%NONLINEAROPERATOR_W Frequency-domain nonlinear operator of the GNLSE.

    if ~isfield(mod,'ssp') || mod.ssp == 1
        NLOP = -1i*gamma*(1 + w/(2*pi*fo)).*fft(((1-fr)*u_t.*abs(u_t).^2) ...
            + fr*dt*u_t.*ifft(hrw.*fft(abs(u_t).^2)));
    else
        NLOP = -1i*gamma*fft(((1-fr)*u_t.*abs(u_t).^2) ...
            + fr*dt*u_t.*ifft(hrw.*fft(abs(u_t).^2)));
    end
end
