% OFC_SIMULATION Simulate a 1050 nm optical frequency comb measurement.
%   Run this script to generate the detected photocurrent for an optical
%   frequency comb centred at 1050 nm with a 100 MHz repetition rate. The
%   model constructs the optical field as a coherent sum of comb lines,
%   computes the direct-detection photocurrent, and applies a brick-wall
%   low-pass filter to emulate a 500 MHz oscilloscope bandwidth.
%
%   You can override the default parameters by defining a struct named
%   OFC_OVERRIDES in the workspace before running the script. Any fields in
%   the struct replace the corresponding defaults.

% Default simulation parameters
defaults = struct( ...
    'CenterWavelengthNm', 1050.0, ...
    'RepetitionRateHz',   100e6, ...
    'CeoFrequencyHz',     20e6, ...
    'NumberOfLines',      25, ...
    'LinewidthHz',        5e6, ...
    'SampleRateHz',       2e9, ...
    'DurationS',          5e-6, ...
    'BandwidthHz',        500e6, ...
    'SaveResults',        true, ...
    'SavePrefix',         'comb_simulation', ...
    'Plot',               false ...
);
if exist('OFC_OVERRIDES', 'var') && isstruct(OFC_OVERRIDES)
    params = apply_overrides(defaults, OFC_OVERRIDES);
else
    params = defaults;
end

numSamples = max(1, round(params.SampleRateHz * params.DurationS));
time = (0:numSamples-1).' / params.SampleRateHz;

indices = (-floor(params.NumberOfLines/2):floor(params.NumberOfLines/2)).';
if mod(numel(indices), 2) == 0
    % Ensure an odd number of comb lines so the central line is included.
    indices = (-params.NumberOfLines/2+1:params.NumberOfLines/2).';
end

envelope = gaussian_envelope(indices, params.LinewidthHz, params.RepetitionRateHz);
if all(envelope == 0)
    error('Linewidth produces zero-valued envelope. Adjust LinewidthHz.');
end
envelope = envelope / sqrt(sum(abs(envelope).^2));

phases = 2*pi*indices*params.CeoFrequencyHz/params.RepetitionRateHz;
frequencies = params.CeoFrequencyHz + indices*params.RepetitionRateHz;

phaseMatrix = 2*pi*(time * frequencies.') + phases.';
opticalField = exp(1i*phaseMatrix) * envelope;

photocurrent = abs(opticalField).^2;
[filteredPhotocurrent, positiveFreqs, positiveSpectrum] = ...
    apply_bandwidth(photocurrent, params.SampleRateHz, params.BandwidthHz);

signal = struct(...
    'time', time,
    'opticalField', opticalField,
    'photocurrent', photocurrent,
    'filteredPhotocurrent', filteredPhotocurrent,
    'frequencies', positiveFreqs,
    'spectrum', positiveSpectrum ...
);

if params.SaveResults
    save([params.SavePrefix, '.mat'], '-struct', 'signal');
end

if params.Plot
    plot_results(signal, params);
end

clear defaults;

function envelope = gaussian_envelope(indices, linewidthHz, repetitionRateHz)
if linewidthHz <= 0
    error('LinewidthHz must be positive.');
end
sigma = linewidthHz / repetitionRateHz;
envelope = exp(-0.5 * (indices / sigma).^2);
end

function [filteredSignal, positiveFreqs, positiveSpectrum] = apply_bandwidth(signal, sampleRateHz, bandwidthHz)
numSamples = numel(signal);
freqResolution = sampleRateHz / numSamples;
freqAxis = (-floor(numSamples/2):ceil(numSamples/2)-1).' * freqResolution;

spectrum = fftshift(fft(signal));
mask = abs(freqAxis) <= bandwidthHz;
filteredSpectrumShifted = spectrum .* mask;
filteredSignal = ifft(ifftshift(filteredSpectrumShifted), 'symmetric');

positiveIndices = 1:floor(numSamples/2)+1;
positiveFreqs = (positiveIndices - 1).' * freqResolution;
positiveSpectrumFull = fft(signal);
positiveSpectrum = positiveSpectrumFull(positiveIndices);
end

function params = apply_overrides(defaults, overrides)
params = defaults;
overrideFields = fieldnames(overrides);
for k = 1:numel(overrideFields)
    name = overrideFields{k};
    if isfield(defaults, name)
        params.(name) = overrides.(name);
    else
        warning('Unknown override field "%s" ignored.', name);
    end
end
end

function plot_results(signal, params)
figure('Name', 'Optical Frequency Comb Simulation', 'Color', 'w');
subplot(2,1,1);
plot(signal.time * 1e6, signal.filteredPhotocurrent, 'LineWidth', 1.1);
xlabel('Time (\mus)');
ylabel('Photocurrent (a.u.)');
title('Filtered Photocurrent vs Time');
grid on;

subplot(2,1,2);
powerSpectrum = abs(signal.spectrum).^2;
plot(signal.frequencies * 1e-6, powerSpectrum, 'LineWidth', 1.1);
xlabel('Frequency (MHz)');
ylabel('Power (a.u.)');
title(sprintf('Spectrum (f_{rep}=%.0f MHz, BW=%.0f MHz)', ...
    params.RepetitionRateHz/1e6, params.BandwidthHz/1e6));
grid on;
end
