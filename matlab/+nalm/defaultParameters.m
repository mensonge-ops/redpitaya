function params = defaultParameters()
%DEFAULTPARAMETERS Default configuration for the 9-shaped NALM laser.
%
%   PARAMS = NALM.DEFAULTPARAMETERS() returns a struct containing all
%   numerical constants required to simulate the all-fibre figure-of-nine
%   ("9"-shaped) nonlinear amplifying loop mirror (NALM) laser described in
%   the user request.  The configuration assumes a 0.6 m Yb401-PM gain
%   fibre, 976 nm single-mode pumping with 1 W maximum power, and a 50:50
%   coupler.  Passive fibres use the dispersion value beta2 = +23 ps^2/km at
%   1030 nm and share the same nonlinear parameters as the doped fibre.
%
%   The struct returned by this function can be customised before calling
%   nalm.runSimulation.
%
%   Fields (selection):
%       lambda_signal      - Signal wavelength (m)
%       lambda_pump        - Pump wavelength (m)
%       pump_power_W       - Pump power coupled into the core (W)
%       kappa              - Power coupling ratio (0.5 for 50/50)
%       beta2_s_m          - Group-velocity dispersion at 1030 nm (s^2/m)
%       dz_max_m           - Maximum longitudinal step for IP-RK4 (m)
%       round_trips        - Number of simulated round trips
%       plot_every         - Plotting cadence (round trips)
%       enable_plots       - Flag controlling live plotting
%       noise_seed         - RNG seed used for the initial noise field
%       initial_noise_W    - RMS power of the initial complex white noise
%       ...
%
%   See also: nalm.runSimulation
%

    c0 = 299792458;                        % speed of light (m/s)
    data = yb.getYb401PMData();            % gain medium parameters

    params.lambda_signal = 1030e-9;
    params.lambda_pump   = 976e-9;
    params.pump_power_W  = 1.0;            % single-mode pump power (W)
    params.kappa         = 0.5;            % 50:50 coupler

    % Dispersion and nonlinearity
    params.beta2_s_m = 23e-27;             % +23 ps^2/km -> 23e-27 s^2/m
    params.n2 = data.n2;
    params.n_core = data.n_core;

    % Effective area and nonlinearity coefficient
    params.A_eff = data.A_eff_m2;
    params.gamma_Wm = 2*pi*params.n2 ./ (params.lambda_signal .* params.A_eff);

    % Gain medium constants
    params.N_total = data.N_total_m3;
    params.tau = data.lifetime_s;
    params.overlap = data.overlap;
    params.sigma_abs_signal = interp1(data.wavelength_nm, data.sigma_abs_m2, ...
        params.lambda_signal * 1e9, 'pchip', 'extrap');
    params.sigma_ems_signal = interp1(data.wavelength_nm, data.sigma_ems_m2, ...
        params.lambda_signal * 1e9, 'pchip', 'extrap');
    params.sigma_abs_pump = interp1(data.wavelength_nm, data.sigma_abs_m2, ...
        params.lambda_pump * 1e9, 'pchip', 'extrap');
    params.sigma_ems_pump = interp1(data.wavelength_nm, data.sigma_ems_m2, ...
        params.lambda_pump * 1e9, 'pchip', 'extrap');
    params.background_loss = 0.002;        % linear loss (1/m)

    params.coupler_phase = -pi/2;          % phase shifter in the loop

    % Fibre lengths (metres)
    params.loop_lengths = struct( ...
        'pre_coupler',    0.2, ...         % passive fibre before the gain fibre
        'gain',           0.6, ...         % Yb401-PM doped fibre
        'wdm',            1.2, ...         % passive fibre / WDM pigtail
        'phase',          1.2, ...         % fibre phase shifter
        'post_coupler',   1.2);            % passive section back to coupler

    params.linear_lengths = struct( ...
        'to_mirror',      1.0, ...         % coupler to fibre mirror (one way)
        'mirror',         1.0);            % fibre section terminated by mirror

    params.output_tap = 0.1;               % tap 10% of power for diagnostics

    % Numerical grid
    params.Nt = 2^12;                      % number of temporal samples
    params.time_window_s = 20e-12;         % 20 ps temporal window
    params.noise_seed = 42;                % RNG seed for reproducibility
    params.initial_noise_W = 1e-6;         % initial white-noise power (W)

    params.dz_max_m = 0.01;                % maximum IP-RK4 longitudinal step

    params.round_trips = 400;              % number of round trips to simulate
    params.plot_every = 10;
    params.enable_plots = true;

    % Derived constants
    params.c0 = c0;
    params.h = 6.62607015e-34;
    params.nu_signal = c0 / params.lambda_signal;
    params.nu_pump = c0 / params.lambda_pump;

    n_g = data.n_core;                     % assume group index ~ refractive
    params.v_g = c0 / n_g;

    total_loop = params.loop_lengths.pre_coupler + params.loop_lengths.gain + ...
        params.loop_lengths.wdm + params.loop_lengths.phase + ...
        params.loop_lengths.post_coupler;
    linear_one_way = params.linear_lengths.to_mirror + params.linear_lengths.mirror;
    params.total_length = total_loop + 2 * linear_one_way;
    params.round_trip_time = params.total_length / params.v_g;

    params.N2_initial = 0.1 * params.N_total;
end
