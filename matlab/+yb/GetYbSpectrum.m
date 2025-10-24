function [sigA, sigE, alpha] = GetYbSpectrum(lam, data)
%GETYBSPECTRUM Interpolate Yb401-PM absorption/emission cross sections.
%
%   [SIGA, SIGE, ALPHA] = GETYBSPECTRUM(LAM) returns the absorption and
%   emission cross sections of the ytterbium-doped Yb401-PM fiber for the
%   wavelengths specified by LAM.  LAM may be given either in metres or in
%   nanometres.  SIGA and SIGE are in units of m^2.  ALPHA is the ratio
%   SIGA ./ SIGE.  Spectral values are obtained by cubic interpolation of
%   the digitised manufacturer data.
%
%   [...] = GETYBSPECTRUM(LAM, DATA) allows supplying a custom data struct
%   as returned by yb.getYb401PMData.
%
%   Example:
%       lam = linspace(900, 1100, 201); % nanometres
%       [sigA, sigE] = yb.GetYbSpectrum(lam);
%
%   See also: yb.getYb401PMData
%

    if nargin < 2 || isempty(data)
        data = yb.getYb401PMData();
    end

    if nargin < 1 || isempty(lam)
        lam = data.wavelength_nm;
    end

    if max(abs(lam)) < 1e-6
        % Input is in metres, convert to nm
        lam_nm = lam * 1e9;
    else
        lam_nm = lam;
    end

    lam_nm = lam_nm(:);

    sigA = interp1(data.wavelength_nm, data.sigma_abs_m2, lam_nm, ...
        'pchip', 'extrap');
    sigE = interp1(data.wavelength_nm, data.sigma_ems_m2, lam_nm, ...
        'pchip', 'extrap');

    alpha = sigA ./ max(sigE, eps);

    if max(abs(lam)) < 1e-6
        % Convert spectra back to the same order as input (row vector when
        % LAM was supplied as a row).
        sigA = reshape(sigA, size(lam));
        sigE = reshape(sigE, size(lam));
        alpha = reshape(alpha, size(lam));
    else
        sigA = reshape(sigA, size(lam));
        sigE = reshape(sigE, size(lam));
        alpha = reshape(alpha, size(lam));
    end
end
