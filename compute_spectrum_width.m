function metrics = compute_spectrum_width(lambda_axis, spectrum)
%COMPUTE_SPECTRUM_WIDTH Estimate spectral width metrics in wavelength units
%   metrics = COMPUTE_SPECTRUM_WIDTH(lambda_axis, spectrum) returns the
%   FWHM in nanometres and the indices of the left/right half-maximum
%   crossings for a given spectral power distribution.  The function is
%   robust to noisy spectra: if the half-maximum crossings cannot be found
%   a width of zero is returned.
%
%   Inputs
%       lambda_axis : wavelength samples (nm)
%       spectrum    : spectral power values (linear units)
%
%   Outputs (structure)
%       metrics.fwhm_nm : estimated FWHM bandwidth (nm)
%       metrics.left_nm : wavelength at the left half-maximum crossing
%       metrics.right_nm: wavelength at the right half-maximum crossing
%       metrics.valid   : logical flag indicating a successful measurement
%
%   The routine normalises the spectrum to its peak and searches for the
%   outermost samples above half the peak value.  Linear interpolation is
%   used to improve the wavelength estimate at the edges.

if isempty(lambda_axis) || isempty(spectrum)
    metrics = empty_metrics();
    return;
end

lambda_axis = lambda_axis(:).';
spectrum = spectrum(:).';

if all(~isfinite(spectrum)) || max(spectrum) <= 0
    metrics = empty_metrics();
    return;
end

spec_norm = spectrum / max(spectrum);

half_level = 0.5;
above = spec_norm >= half_level;
idx = find(above);

if numel(idx) < 2
    metrics = empty_metrics();
    return;
end

left_idx = idx(1);
right_idx = idx(end);

% Linear interpolation for improved accuracy
left_nm = interpolate_edge(lambda_axis, spec_norm, left_idx, half_level, -1);
right_nm = interpolate_edge(lambda_axis, spec_norm, right_idx, half_level, +1);

metrics.fwhm_nm = abs(right_nm - left_nm);
metrics.left_nm = left_nm;
metrics.right_nm = right_nm;
metrics.valid = true;

end

function val = interpolate_edge(lambda_axis, spec_norm, idx, level, direction)
%INTERPOLATE_EDGE Linear interpolation around the half-maximum crossing
next_idx = idx + direction;
if next_idx < 1 || next_idx > numel(spec_norm)
    val = lambda_axis(idx);
    return;
end

x1 = spec_norm(idx);
x2 = spec_norm(next_idx);

if x1 == x2
    val = lambda_axis(idx);
    return;
end

lambda1 = lambda_axis(idx);
lambda2 = lambda_axis(next_idx);

val = lambda1 + (level - x1) * (lambda2 - lambda1) / (x2 - x1);
end

function metrics = empty_metrics()
metrics = struct('fwhm_nm', 0, 'left_nm', NaN, 'right_nm', NaN, 'valid', false);
end
