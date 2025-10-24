function params = GetYb401PMGainParams()
%GETYB401PMGAINPARAMS Gain model parameters for Nufern Yb401-PM fiber.
%   PARAMS = GETYB401PMGAINPARAMS() returns a struct that contains all
%   material constants required by the Yb-doped fiber gain model used in the
%   NALM laser simulator.  The fields are organised so that they can be fed
%   into a rate-equation solver or an interaction-picture RK4 propagator.
%
%   The struct contains the following fields:
%       crossSections  - struct with wavelength grid and sigma data
%       tau            - upper-state lifetime (s)
%       N_total        - dopant concentration (m^-3)
%       A_core         - core area (m^2)
%       n_core         - refractive index of the core at 1030 nm
%       beta2          - group-velocity dispersion in s^2/m
%       pump_wavelength- pump wavelength in meters
%       pump_sigma_abs - absorption cross section at pump wavelength
%       pump_sigma_em  - emission cross section at pump wavelength
%       signal_wavelength - signal wavelength (m)
%       signal_sigma_abs  - absorption cross section at signal wavelength
%       signal_sigma_em   - emission cross section at signal wavelength
%       L                - physical fiber length (m)
%
%   The numbers are consistent with Nufern's typical data sheet: the core
%   diameter is 6.5 um (A_core = pi * (3.25 um)^2), the numerical aperture is
%   0.12 and the Yb3+ concentration corresponds to 1.05e26 ions/m^3.  The
%   radiative lifetime is taken as 0.85 ms.  The group-velocity dispersion is
%   +23 ps^2/km at 1030 nm, as specified by the user request.
%
%   The helper interpolates the wavelength-dependent cross sections from the
%   tabulated data returned by LOADYBCROSSSECTIONS.
%
%   See also LOADYBCROSSSECTIONS.

    data = LoadYbCrossSections();

    params = struct();
    params.crossSections = data;
    params.tau = 0.85e-3;                 % s
    params.N_total = 1.05e26;             % m^-3
    core_radius = 3.25e-6;                % m (half of 6.5 um)
    params.A_core = pi * core_radius^2;   % m^2
    params.n_core = 1.450;
    params.beta2 = 23e-27;                % s^2/m
    params.L = 0.6;                       % m fiber length

    params.pump_wavelength = 976e-9;      % m
    params.signal_wavelength = 1030e-9;   % m

    params.pump_sigma_abs = interp1(data.wavelength_nm, data.sigma_abs, ...
        params.pump_wavelength * 1e9, 'linear');
    params.pump_sigma_em = interp1(data.wavelength_nm, data.sigma_em, ...
        params.pump_wavelength * 1e9, 'linear');
    params.signal_sigma_abs = interp1(data.wavelength_nm, data.sigma_abs, ...
        params.signal_wavelength * 1e9, 'linear');
    params.signal_sigma_em = interp1(data.wavelength_nm, data.sigma_em, ...
        params.signal_wavelength * 1e9, 'linear');

    % Sanity check for interpolation: ensure we have values for pump and signal
    if any(isnan([params.pump_sigma_abs, params.pump_sigma_em, ...
            params.signal_sigma_abs, params.signal_sigma_em]))
        error('GetYb401PMGainParams:OutOfRange', ...
            'Requested wavelengths are outside the tabulated data range.');
    end
end
