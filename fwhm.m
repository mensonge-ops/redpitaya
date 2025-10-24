function [width, I_l, I_r] = fwhm(x)
%FWHM Compute the discrete full width at half maximum of vector X.
%   [WIDTH, I_L, I_R] = FWHM(X) returns the number of samples enclosed
%   between the two half-maximum crossings of the waveform X.  The helper
%   tolerates non-ideal traces: if the half-maximum cannot be located the
%   function returns NaN for the width and indices.

x = x(:).';

[peak, ind_peak] = max(x);
if isempty(ind_peak) || peak <= 0
    width = NaN;
    I_l = NaN;
    I_r = NaN;
    return;
end

half_peak = peak / 2;

x_l = x(1:ind_peak);
x_r = x(ind_peak:end);

left_idx = find(fliplr(x_l) <= half_peak, 1, 'first');
right_idx = find(x_r <= half_peak, 1, 'first');

if isempty(left_idx) || isempty(right_idx)
    width = NaN;
    I_l = NaN;
    I_r = NaN;
    return;
end

I_l = ind_peak - left_idx + 1;
I_r = ind_peak + right_idx - 1;
width = I_r - I_l;
end
