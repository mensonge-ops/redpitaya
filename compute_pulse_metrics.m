function metrics = compute_pulse_metrics(u, t, dt, freq, c)
%COMPUTE_PULSE_METRICS  Derive temporal and spectral figures of merit.
%   metrics = COMPUTE_PULSE_METRICS(u, t, dt, freq, c) analyses the
%   complex field u and returns a struct with pulse energy, peak power,
%   temporal FWHM, spectral FWHM (in nm and THz) and RMS bandwidth.  The
%   frequency axis must correspond to the FFT definition used by the caller
%   (freq in THz, centred around the carrier).  The speed of light c must
%   be in nm/ps.
%
%   The helper relies on the legacy fwhm.m routine (sample-count FWHM) and
%   augments it with axis-aware FWHM calculations for the spectral domain.

intensity = abs(u).^2;
energy = dt * sum(intensity);
peak_power = max(intensity);

[width_samples, idx_left, idx_right] = fwhm(intensity);
if isnan(width_samples) || isinf(width_samples)
    width_samples = 0;
end

metrics.energy = energy;
metrics.peak_power = peak_power;
metrics.time_fwhm = width_samples * dt;
metrics.time_window = [t(max(idx_left, 1)), t(min(idx_right, numel(t)))];

% Frequency-domain analysis
spectrum = abs(fftshift(fft(u))).^2;
% Convert to spectral density versus wavelength (nm)
lambda_axis = c ./ freq;
lambda_axis(isnan(lambda_axis)) = 0;
lambda_axis(isinf(lambda_axis)) = 0;

spec_lambda = spectrum ./ (lambda_axis.^2 + eps);
if max(spec_lambda) > 0
    spec_lambda = spec_lambda ./ max(spec_lambda);
end

% Sort axes to make them monotonic for FWHM computation
[lambda_sorted, sort_idx] = sort(lambda_axis);
spec_sorted = spec_lambda(sort_idx);

[lambda_fwhm, lambda_left, lambda_right] = fwhm_axis(lambda_sorted, spec_sorted);
metrics.lambda_fwhm = lambda_fwhm;
metrics.lambda_window = [lambda_left, lambda_right];

% Frequency FWHM (THz)
freq_axis = freq(sort_idx);
[nu_fwhm, nu_left, nu_right] = fwhm_axis(freq_axis, spec_sorted);
metrics.freq_fwhm = nu_fwhm;
metrics.freq_window = [nu_left, nu_right];

% RMS bandwidths (frequency and wavelength)
if sum(spec_sorted) > 0
    power_norm = spec_sorted ./ sum(spec_sorted);
    lambda_mean = sum(lambda_sorted .* power_norm);
    freq_mean = sum(freq_axis .* power_norm);
    metrics.lambda_rms = sqrt(sum(((lambda_sorted - lambda_mean).^2) .* power_norm));
    metrics.freq_rms = sqrt(sum(((freq_axis - freq_mean).^2) .* power_norm));
else
    metrics.lambda_rms = 0;
    metrics.freq_rms = 0;
end

metrics.spectrum_lambda = spec_lambda;
metrics.lambda_axis = lambda_axis;
metrics.freq_axis = freq_axis;

end
