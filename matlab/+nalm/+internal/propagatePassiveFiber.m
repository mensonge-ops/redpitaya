function field = propagatePassiveFiber(field, length_m, params, grid)
%PROPAGATEPASSIVEFIBER Propagate through a passive fibre section.

    if length_m <= 0
        return;
    end

    steps = max(1, ceil(length_m / params.dz_max_m));
    dz = length_m / steps;

    beta2 = params.beta2_s_m;
    loss = params.background_loss;

    omega = 2*pi * grid.f;          % match FFT ordering
    omega = ifftshift(omega);       % convert to standard FFT order

    lin_op = 0.5 * ( -loss ) + 0.5i * beta2 .* (omega.^2);

    field = nalm.internal.ip_rk4(field, dz, steps, lin_op, params.gamma_Wm);
end
