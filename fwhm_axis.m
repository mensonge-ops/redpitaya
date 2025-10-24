function [width, left_val, right_val] = fwhm_axis(axis, values)
%FWHM_AXIS  Compute full-width at half maximum on an arbitrary axis.
%   [width, left_val, right_val] = FWHM_AXIS(axis, values) returns the
%   width measured between the first pair of points at which the signal
%   drops below half of its maximum.  The axis vector must be monotonic but
%   can be increasing or decreasing.  If the half-maximum cannot be
%   identified, width is set to zero and the edge values return NaN.

if isempty(axis) || isempty(values)
    width = 0;
    left_val = NaN;
    right_val = NaN;
    return;
end

[axis, sort_idx] = sort(axis(:));
values = values(:);
values = values(sort_idx);

[peak, peak_idx] = max(values);
if peak <= 0
    width = 0;
    left_val = NaN;
    right_val = NaN;
    return;
end

half_peak = peak/2;
left_region = values(1:peak_idx);
right_region = values(peak_idx:end);

left_idx = find(left_region <= half_peak, 1, 'last');
right_idx = find(right_region <= half_peak, 1, 'first');

if isempty(left_idx) || isempty(right_idx)
    width = 0;
    left_val = NaN;
    right_val = NaN;
    return;
end

left_idx = max(left_idx, 1);
right_idx = peak_idx - 1 + right_idx;

left_val = axis(left_idx);
right_val = axis(right_idx);
width = abs(right_val - left_val);

end
