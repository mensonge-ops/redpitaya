function [width_nm, left_idx, right_idx] = spectral_fwhm(power_norm, lambda)
%SPECTRAL_FWHM Compute the full width at half maximum of a spectrum.
%
%   power_norm : normalised spectral power density (row or column vector)
%   lambda     : wavelength vector (same size as power_norm)
%
% Returns the FWHM in nm together with the indices of the left/right
% half-maximum points. If the FWHM cannot be determined the function
% returns NaN.

if isempty(power_norm) || isempty(lambda)
    width_nm = NaN;
    left_idx = NaN;
    right_idx = NaN;
    return;
end

power_norm = power_norm(:)';
lambda = lambda(:)';

[max_val, max_idx] = max(power_norm);
if max_val <= 0
    width_nm = NaN;
    left_idx = NaN;
    right_idx = NaN;
    return;
end

half_val = 0.5*max_val;
left_segment = power_norm(1:max_idx);
right_segment = power_norm(max_idx:end);

left_cross = find(left_segment <= half_val, 1, 'last');
right_cross = find(right_segment <= half_val, 1, 'first');

if isempty(left_cross) || isempty(right_cross)
    width_nm = NaN;
    left_idx = NaN;
    right_idx = NaN;
    return;
end

left_idx = left_cross;
right_idx = max_idx + right_cross - 1;

width_nm = abs(lambda(right_idx) - lambda(left_idx));
end
