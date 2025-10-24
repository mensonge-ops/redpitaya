function [u_out, Plotdata] = apply_cfbg(u_in, cfbg, fo, df, c)
%APPLY_CFBG   Apply a Gaussian-profiled chirped fibre Bragg grating
%   [u_out, Plotdata] = APPLY_CFBG(u_in, cfbg, fo, df, c) multiplies the
%   field u_in by the complex transfer function of a CFBG described by
%   the parameter struct cfbg.  The grating provides a Gaussian spectral
%   reflectivity with peak power reflectivity cfbg.reflectivity and adds a
%   quadratic spectral phase term determined by the specified dispersion
%   coefficient (group-delay slope in ps/nm).
%
%   The cfbg structure must provide:
%       lambda_c    - central wavelength (nm)
%       bandwidth   - FWHM bandwidth (nm)
%       reflectivity- peak power reflectivity (0..1)
%       dispersion  - group delay slope (ps/nm)
%       fc          - central frequency (THz)
%       beta2       - second-order phase coefficient (ps^2)
%
%   Plotdata holds replicated spectra/time traces so the caller can append
%   them to the master Plotdata diagnostics matrix (mimicking other fibre
%   sections in the code base).
%
%   The implementation assumes the signal is defined around fo (THz) with
%   a sampling step df (THz) and that the speed of light is provided in
%   nm/ps via c.

Ui = fft(u_in);
N = numel(Ui);

% Frequency grid around the carrier
freq = (-(N/2)*df:df:(N/2-1)*df) + fo;          % THz
freq(abs(freq) < eps) = eps;                    % avoid division by zero
omega = 2*pi*freq;                               % rad/ps
omega0 = 2*pi*cfbg.fc;                           % rad/ps
lambda = c./freq;                                % nm

% Gaussian reflectivity profile
sigma = cfbg.bandwidth / (2*sqrt(2*log(2)));      % nm
amp = sqrt(cfbg.reflectivity) * ...
    exp(-0.5 * ((lambda - cfbg.lambda_c) ./ sigma).^2);

% Quadratic spectral phase from the dispersion coefficient
phase = 0.5 * cfbg.beta2 * (omega - omega0).^2;   % rad

transfer = amp .* exp(1i * phase);

% Apply transfer function (spectrum handled in centred form)
Ui_shift = fftshift(Ui);
Uo_shift = Ui_shift .* transfer;
Uo = ifftshift(Uo_shift);

u_out = ifft(Uo);

% Diagnostics for downstream plots
Plotdata.ufft = repmat(abs(fftshift(fft(u_out))), 20, 1);
Plotdata.u = repmat(u_out, 20, 1);
Plotdata.transfer = transfer;

end
