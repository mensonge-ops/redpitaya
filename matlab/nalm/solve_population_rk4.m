function [N2, gain_profile] = solve_population_rk4(field, params, grid, pumpPower)
%SOLVE_POPULATION_RK4 Evolve the excited-state population with RK4.
%   [N2, GAIN_PROFILE] = SOLVE_POPULATION_RK4(FIELD, PARAMS, GRID, PUMPPOWER)
%   integrates the rate equation for the upper state population N2 of the
%   Yb-doped gain fiber using a fourth-order Runge-Kutta scheme.  FIELD is the
%   complex temporal envelope of the signal, PARAMS is the struct returned by
%   GETYB401PMGAINPARAMS, GRID provides the temporal spacing, and PUMPPOWER is
%   the launched pump power in watts.  The function returns the time-dependent
%   population N2 (ions/m^3) and the corresponding gain coefficient profile
%   GAIN_PROFILE (1/m) evaluated at the signal wavelength.

    h = 6.62607015e-34;          % Planck constant (J*s)
    c = 299792458;               % speed of light (m/s)

    intensity = abs(field).^2 / params.A_core;             % W/m^2
    photon_flux_signal = intensity * params.signal_wavelength / (h * c);

    pump_flux = pumpPower / (h * c / params.pump_wavelength) / params.A_core;
    pump_abs_rate = pump_flux * params.pump_sigma_abs;
    pump_em_rate = pump_flux * params.pump_sigma_em;

    % Steady-state inversion with negligible signal (used as initial value)
    N2_ss = params.N_total * pump_abs_rate / ...
        (pump_abs_rate + pump_em_rate + 1/params.tau);

    N2 = zeros(size(field));
    current = N2_ss;

    for idx = 1:numel(field)
        Phi_s = photon_flux_signal(idx);
        function dN = rhs(pop)
            ground = params.N_total - pop;
            stimulated = Phi_s * (params.signal_sigma_em * pop + ...
                params.signal_sigma_abs * ground);
            dN = pump_abs_rate * ground - pump_em_rate * pop ...
                - stimulated - pop / params.tau;
        end

        k1 = rhs(current);
        k2 = rhs(current + 0.5 * grid.dt * k1);
        k3 = rhs(current + 0.5 * grid.dt * k2);
        k4 = rhs(current + grid.dt * k3);

        updated = current + grid.dt * (k1 + 2*k2 + 2*k3 + k4) / 6;
        current = min(max(updated, 0), params.N_total);
        N2(idx) = current;
    end

    gain_profile = params.signal_sigma_em .* N2 - ...
        params.signal_sigma_abs .* (params.N_total - N2);
end
