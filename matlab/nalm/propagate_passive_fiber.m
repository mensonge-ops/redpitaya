function field = propagate_passive_fiber(field, length_m, fiber, grid)
%PROPAGATE_PASSIVE_FIBER Propagate a field through passive fiber.
%   FIELD = PROPAGATE_PASSIVE_FIBER(FIELD, LENGTH_M, FIBER, GRID) advances
%   the complex temporal envelope FIELD through a section of passive fiber
%   of length LENGTH_M using an interaction-picture fourth-order Runge-Kutta
%   scheme (IP-RK4).  The FIBER struct must define the dispersion coefficient
%   beta2 (s^2/m), the nonlinear coefficient gamma (1/W/m), the power loss
%   coefficient alpha (1/m) and the longitudinal step size dz (m).  GRID is a
%   struct with the fields t (time grid), dt (time spacing) and w (angular
%   frequency grid) used for the Fourier transforms.
%
%   The implementation follows the algorithm described in Hult, J. Lightwave
%   Technol. 25(12), 3770-3775 (2007).  Linear dispersion is applied in the
%   frequency domain with a half-step before and after each nonlinear RK4
%   update.

    arguments
        field (:,1) double
        length_m (1,1) double {mustBeNonnegative}
        fiber.beta2 (1,1) double
        fiber.gamma (1,1) double
        fiber.alpha (1,1) double
        fiber.dz (1,1) double {mustBePositive}
        grid struct
    end

    if length_m == 0 || all(field == 0)
        return;
    end

    nsteps = max(1, ceil(length_m / fiber.dz));
    dz = length_m / nsteps;

    half_linear = exp(1i * 0.5 * fiber.beta2 * (grid.w.^2) * dz);

    for step = 1:nsteps
        field = ifft(half_linear .* fft(field));
        field = rk4_nonlinear_step(field, dz, fiber.gamma, fiber.alpha);
        field = ifft(half_linear .* fft(field));
    end
end

function field = rk4_nonlinear_step(field, dz, gamma, alpha)
    function rhs = nonlinear_term(u)
        rhs = 1i * gamma * abs(u).^2 .* u - 0.5 * alpha * u;
    end

    k1 = dz * nonlinear_term(field);
    k2 = dz * nonlinear_term(field + 0.5 * k1);
    k3 = dz * nonlinear_term(field + 0.5 * k2);
    k4 = dz * nonlinear_term(field + k3);

    field = field + (k1 + 2*k2 + 2*k3 + k4) / 6;
end
