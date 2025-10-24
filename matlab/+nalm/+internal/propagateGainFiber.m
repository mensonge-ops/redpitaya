function [field, state, metrics] = propagateGainFiber(field, state, params, grid)
%PROPAGATEGAINFIBER Propagate through the Yb401-PM gain fibre.

    length_m = params.loop_lengths.gain;
    steps = max(1, ceil(length_m / params.dz_max_m));
    dz = length_m / steps;

    % Gain coefficient derived from current inversion
    g0 = params.overlap * (params.sigma_ems_signal * state.N2 - ...
        params.sigma_abs_signal * (params.N_total - state.N2));
    loss = params.background_loss;

    omega = 2*pi * grid.f;
    omega = ifftshift(omega);
    lin_op = 0.5 * (g0 - loss) + 0.5i * params.beta2_s_m .* (omega.^2);

    field = nalm.internal.ip_rk4(field, dz, steps, lin_op, params.gamma_Wm);

    metrics.gain_per_m = g0 - loss;
end
