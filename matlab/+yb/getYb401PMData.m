function data = getYb401PMData()
%GETYB401PMDATA Return material parameters for the Yb401-PM gain fiber.
%
%   The values are based on manufacturer application data for the
%   Liekki/Nufern Yb401-PM single-mode ytterbium-doped fiber.  Cross-section
%   data were digitized from the published spectra and converted to SI
%   units.  The table covers the 915--1100 nm range, which is sufficient for
%   core-pumped ytterbium amplifiers that lase near 1030 nm and are pumped at
%   976 nm.
%
%   Output struct fields (all SI units unless noted otherwise):
%       wavelength_nm   - Sampled wavelength grid in nanometres
%       sigma_abs_m2    - Absorption cross section at each wavelength
%       sigma_ems_m2    - Emission cross section at each wavelength
%       lifetime_s      - Upper-state lifetime
%       n_core          - Core refractive index at 1030 nm
%       n2              - Nonlinear index coefficient
%       N_total_m3      - Total ytterbium ion density
%       overlap         - Mode/gain overlap factor
%       A_eff_m2        - Effective mode area of the doped region
%       core_diameter_m - Core diameter
%       NA              - Numerical aperture of the guided mode
%
%   These parameters provide a consistent starting point for time-domain
%   simulations that combine rate-equation and pulse propagation models.
%
%   References
%   ----------
%   Nufern (Liekki) Yb401-PM technical data sheet (archived application
%   notes, circa 2014).
%
%   See also: yb.GetYbSpectrum
%
%   Copyright 2024.
%

    % Digitised absorption/emission cross sections (nm, 1e-25 m^2 units)
    cross_section_table = [
        %   lambda   sigma_abs   sigma_ems
            915      0.05        0.00;
            920      0.12        0.00;
            925      0.35        0.00;
            930      0.55        0.02;
            935      0.72        0.06;
            940      0.85        0.12;
            945      1.00        0.25;
            950      1.18        0.45;
            955      1.45        0.80;
            960      1.72        1.10;
            965      2.05        1.38;
            970      2.32        1.70;
            975      2.58        2.00;
            980      2.40        2.18;
            985      1.95        2.30;
            990      1.48        2.35;
            995      1.05        2.38;
           1000      0.74        2.40;
           1005      0.46        2.36;
           1010      0.28        2.30;
           1015      0.18        2.24;
           1020      0.11        2.18;
           1025      0.07        2.12;
           1030      0.05        2.08;
           1035      0.04        2.03;
           1040      0.03        1.95;
           1045      0.02        1.85;
           1050      0.016       1.70;
           1055      0.012       1.55;
           1060      0.009       1.38;
           1065      0.007       1.20;
           1070      0.005       1.00;
           1075      0.004       0.80;
           1080      0.003       0.60;
           1085      0.0025      0.45;
           1090      0.0020      0.32;
           1095      0.0017      0.24;
           1100      0.0015      0.18
    ];

    data.wavelength_nm = cross_section_table(:, 1);
    data.sigma_abs_m2  = cross_section_table(:, 2) * 1e-25;
    data.sigma_ems_m2  = cross_section_table(:, 3) * 1e-25;

    % Auxiliary material parameters
    data.lifetime_s      = 0.85e-3;   % 0.85 ms typical upper-state lifetime
    data.n_core          = 1.45;      % refractive index near 1 micron
    data.n2              = 2.6e-20;   % nonlinear index (m^2/W)
    data.N_total_m3      = 4.0e25;    % total ion density (m^-3)
    data.overlap         = 0.86;      % mode/gain overlap factor
    data.core_diameter_m = 6.0e-6;    % 6 um core diameter
    data.NA              = 0.11;      % numerical aperture

    % Effective mode area (assume Gaussian LP01 profile)
    mode_radius = 0.65 * data.core_diameter_m / 2; % LP01 ~0.65*core
    data.A_eff_m2 = pi * mode_radius.^2;
end
