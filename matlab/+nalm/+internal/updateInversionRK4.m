function [N2_next, rhs_val] = updateInversionRK4(N2, params, pump_flux, signal_power_W)
%UPDATEINVERSIONRK4 Update upper-state population with RK4.
%
%   signal_power_W is the average power circulating in the gain fibre for
%   the current round trip.  pump_flux is the photon flux of the pump in
%   photons/(m^2*s).

    dt = params.round_trip_time;
    hnu_s = params.h * params.nu_signal;

    signal_flux = signal_power_W / (params.A_eff * hnu_s);

    function dN = rhs(pop)
        absorption = pump_flux * params.sigma_abs_pump * (params.N_total - pop);
        stimulated_pump = pump_flux * params.sigma_ems_pump * pop;
        stimulated_signal = signal_flux * params.sigma_ems_signal * pop;
        spontaneous = pop / params.tau;
        dN = absorption - stimulated_pump - stimulated_signal - spontaneous;
    end

    k1 = dt * rhs(N2);
    k2 = dt * rhs(N2 + 0.5 * k1);
    k3 = dt * rhs(N2 + 0.5 * k2);
    k4 = dt * rhs(N2 + k3);

    N2_next = N2 + (k1 + 2*k2 + 2*k3 + k4) / 6;
    N2_next = max(0, min(params.N_total, N2_next));

    rhs_val = rhs(N2_next);
end
