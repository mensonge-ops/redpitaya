function metrics = compute_locking_metrics(u_field, dt, f, fo, c)
%COMPUTE_LOCKING_METRICS  Derive pulse metrics for lock-state detection.
%   metrics = COMPUTE_LOCKING_METRICS(u_field, dt, f, fo, c) calculates
%   energy, peak power, temporal FWHM, spectral FWHM (nm) and RMS frequency
%   shift for the complex field u_field sampled at dt (ps).  The frequency
%   vector f (THz) is centred around zero, fo is the carrier frequency (THz)
%   and c is the speed of light in nm/ps.

intensity = abs(u_field).^2;
metrics.energy_pJ = dt * sum(intensity);          % pulse energy (pJ)
metrics.peak_power_W = max(intensity);            % peak power (W)

% Temporal FWHM in ps
[width_samples, ~, ~] = fwhm(intensity);
metrics.fwhm_time_ps = width_samples * dt;

% Chirp estimate using instantaneous frequency deviation
phase = unwrap(angle(u_field));
if numel(phase) > 1
    inst_freq = -diff(phase) / (2*pi*dt);          % THz
    metrics.chirp_rms_THz = rms(inst_freq);
else
    metrics.chirp_rms_THz = 0;
end

% Spectral FWHM using wavelength axis
spec = fftshift(abs(fft(u_field)).^2);
spec = spec / (max(spec) + eps);
lambda_axis = c ./ (f + fo);
metrics.spectral_width_nm = measure_spectral_width(lambda_axis, spec);
metrics.lambda_axis = lambda_axis;                % retain for inspection
metrics.spectrum = spec;

end
