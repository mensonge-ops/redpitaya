function width_nm = measure_spectral_width(lambda_axis, spectrum)
%MEASURE_SPECTRAL_WIDTH  Estimate the FWHM bandwidth in nanometres.
%   width_nm = MEASURE_SPECTRAL_WIDTH(lambda_axis, spectrum) normalises the
%   spectral power array "spectrum" to its maximum and measures the full
%   width at half maximum along the wavelength vector lambda_axis.  The
%   function assumes lambda_axis is monotonically increasing or decreasing.
%
%   If multiple peaks exist, the outermost half-maximum crossings are used.
%   When the spectrum never crosses the half-maximum level the returned
%   bandwidth is zero.

if isempty(lambda_axis) || isempty(spectrum)
    width_nm = 0;
    return;
end

% Ensure column vectors and sort by wavelength to guarantee monotonicity.
lambda_axis = lambda_axis(:);
spectrum = spectrum(:);
[lambda_sorted, idx] = sort(lambda_axis, 'ascend');
spectrum_sorted = spectrum(idx);

if max(spectrum_sorted) <= 0
    width_nm = 0;
    return;
end

spec_norm = spectrum_sorted ./ max(spectrum_sorted);
half_level = 0.5;
above_half = spec_norm >= half_level;

if ~any(above_half)
    width_nm = 0;
    return;
end

first_idx = find(above_half, 1, 'first');
last_idx = find(above_half, 1, 'last');

width_nm = lambda_sorted(last_idx) - lambda_sorted(first_idx);
width_nm = abs(width_nm);

end
