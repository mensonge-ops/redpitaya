function field = ip_rk4(field, dz, steps, lin_op, gamma)
%IP_RK4 Interaction-picture RK4 propagator for the cubic NLSE.
%
%   FIELD = IP_RK4(FIELD, DZ, STEPS, LIN_OP, GAMMA) propagates the complex
%   envelope FIELD over a distance DZ * STEPS.  LIN_OP must match the
%   ordering of FFT(FIELD) and typically combines dispersion, gain and loss.
%
%   The algorithm follows the description in G. P. Agrawal, "Nonlinear
%   Fiber Optics", 5th ed., Section 2.4.  The code is vectorised and works
%   for row-vector fields.

    if steps <= 0
        return;
    end

    lin_half = exp(lin_op * (dz / 2));
    lin_full = exp(lin_op * dz);

    for idx = 1:steps
        A1 = ifft(lin_half .* fft(field, [], 2), [], 2);
        k1 = dz * nonlinearTerm(A1, gamma);

        A2 = ifft(lin_half .* fft(field + 0.5 * k1, [], 2), [], 2);
        k2 = dz * nonlinearTerm(A2, gamma);

        A3 = ifft(lin_half .* fft(field + 0.5 * k2, [], 2), [], 2);
        k3 = dz * nonlinearTerm(A3, gamma);

        A4 = ifft(lin_full .* fft(field + k3, [], 2), [], 2);
        k4 = dz * nonlinearTerm(A4, gamma);

        field = ifft(lin_half .* fft(field + ...
            (k1 + 2*k2 + 2*k3 + k4) / 6, [], 2), [], 2);
    end
end

function term = nonlinearTerm(field, gamma)
    term = 1i * gamma .* abs(field).^2 .* field;
end
