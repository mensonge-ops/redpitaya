function [field_next, state, metrics] = roundTrip(field_in, state, params, grid)
%ROUNDTRIP Perform one full cavity round trip.

    % Split at the coupler into CW/CCW components (vacuum on auxiliary port)
    [cw, ccw] = nalm.internal.couplerSplit(field_in, 0, params.kappa);

    % Pre-gain passive section
    cw = nalm.internal.propagatePassiveFiber(cw, params.loop_lengths.pre_coupler, params, grid);
    ccw = nalm.internal.propagatePassiveFiber(ccw, params.loop_lengths.pre_coupler, params, grid);

    % Gain fibre (propagate separately to accumulate SPM)
    [cw, state, gain_metrics_cw] = nalm.internal.propagateGainFiber(cw, state, params, grid);
    [ccw, ~, gain_metrics_ccw] = nalm.internal.propagateGainFiber(ccw, state, params, grid);

    % WDM passive segment
    cw = nalm.internal.propagatePassiveFiber(cw, params.loop_lengths.wdm, params, grid);
    ccw = nalm.internal.propagatePassiveFiber(ccw, params.loop_lengths.wdm, params, grid);

    % Phase bias (-pi/2 applied uniformly)
    cw = cw * exp(1i * params.coupler_phase / 2);
    ccw = ccw * exp(1i * params.coupler_phase / 2);

    % Additional passive fibre back to the coupler
    cw = nalm.internal.propagatePassiveFiber(cw, params.loop_lengths.post_coupler, params, grid);
    ccw = nalm.internal.propagatePassiveFiber(ccw, params.loop_lengths.post_coupler, params, grid);

    % Recombine at the coupler
    [reflected, transmitted] = nalm.internal.couplerCombine(cw, ccw, params.kappa);

    % Linear arm propagation (two-way)
    forward_linear = nalm.internal.propagatePassiveFiber(reflected, params.linear_lengths.to_mirror, params, grid);
    tap_field = sqrt(params.output_tap) * forward_linear;
    forward_linear = sqrt(max(0, 1 - params.output_tap)) * forward_linear;

    forward_linear = nalm.internal.propagatePassiveFiber(forward_linear, params.linear_lengths.mirror, params, grid);
    reflected_linear = -forward_linear; % fibre Bragg mirror (pi phase shift)

    backward_linear = nalm.internal.propagatePassiveFiber(reflected_linear, params.linear_lengths.mirror, params, grid);
    backward_linear = nalm.internal.propagatePassiveFiber(backward_linear, params.linear_lengths.to_mirror, params, grid);

    field_next = backward_linear;
    state.linear_field = backward_linear;

    % Update gain dynamics based on total energy inside the doped fibre
    energy_cw = trapz(grid.t, abs(cw).^2);
    energy_ccw = trapz(grid.t, abs(ccw).^2);
    avg_power = (energy_cw + energy_ccw) / params.round_trip_time;

    [state.N2, ~] = nalm.internal.updateInversionRK4(state.N2, params, state.pump_flux, avg_power);

    % Diagnostics
    metrics.output_pulse = transmitted;
    metrics.output_spectrum = abs(fftshift(fft(transmitted))).^2;
    metrics.output_energy = trapz(grid.t, abs(transmitted).^2);
    metrics.loop_energy = energy_cw + energy_ccw;
    metrics.tap_energy = trapz(grid.t, abs(tap_field).^2);
    metrics.inversion_ratio = state.N2 / params.N_total;
    metrics.gain_per_m = 0.5 * (gain_metrics_cw.gain_per_m + gain_metrics_ccw.gain_per_m);
end
