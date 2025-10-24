function params = default_yb401_params()
%DEFAULT_YB401_PARAMS  Return baseline parameters for the Yb401-PM NALM
%   params = DEFAULT_YB401_PARAMS() builds a struct describing the cavity,
%   gain media, CFBG, and numerical grid used by the Yb401-PM based NALM
%   mode-locking simulations.  The values are derived from typical fibre and
%   component data sheets and tuned to favour broad-band pulses near 1030 nm.
%
%   The returned struct contains:
%       params.constants   - numerical constants (speed of light)
%       params.pulse       - input pulse specification
%       params.fibre       - base fibre properties
%       params.sections    - lengths and gains of cavity sections
%       params.cfbg        - chirped fibre Bragg grating parameters
%       params.couplers    - NALM and output coupler ratios
%       params.grid        - temporal/frequency grid configuration
%
%   These defaults can be modified prior to invoking RUN_YB401_NALM.

params.constants.c = 299792.458;            % speed of light (nm/ps)

% Input pulse slightly above the fundamental soliton order with a few-ps
% duration to ease self-starting while allowing significant spectral
% broadening once gain builds up.
params.pulse.N2 = 1.35^2;                   % soliton order squared
params.pulse.tfwhm = 3.5;                   % FWHM (ps)
params.pulse.lambda = 1030;                 % central wavelength (nm)
params.pulse.noise_level = 5e-3;            % relative Gaussian noise
params.pulse.random_seed = 0;               % deterministic seed for repeatability

% Yb401-PM fibre properties (effective area from 6 um MFD, D ~= 20 ps/nm/km)
params.fibre.Aeff = 28.3;                   % effective mode area (um^2)
params.fibre.n2 = 26;                       % Kerr coefficient (10^-16 cm^2/W)
params.fibre.alpha_dB_per_m = 0.25;         % loss coefficient (dB/m)
params.fibre.beta2 = -11.26;                % ps^2/km (converted from D)
params.fibre.beta3 = 0.1;                   % ps^3/km
params.fibre.include_raman = false;
params.fibre.include_ssp = false;

% Section lengths (km) roughly matching a compact laboratory cavity.  These
% can be altered freely by callers or the parameter scan utility.
sections.smf_link = 0.0005;                 % fibre before main gain (0.5 m)
sections.amf_main = struct('L', 0.0006, ... % main gain fibre (0.6 m)
                           'gssdB', 38, ...% small-signal gain (dB)
                           'PsatdBm', 33, ...
                           'bandwidth_nm', 8);
sections.smf_output = 0.0006;               % fibre between gain and NALM (0.6 m)
sections.smf_pre = 0.0006;                  % first NALM passive arm segment (0.6 m)
sections.amf_nalm = struct('L', 0.0004, ... % NALM gain fibre (0.4 m)
                           'gssdB', 27, ...
                           'PsatdBm', 32, ...
                           'bandwidth_nm', 6);
sections.smf_post = 0.0007;                 % second NALM passive arm segment (0.7 m)
sections.smf_linear = 0.0004;               % fibre to output branch (0.4 m)
params.sections = sections;

% Chirped fibre Bragg grating (linear arm reflector)
params.cfbg.lambda_c = params.pulse.lambda;
params.cfbg.bandwidth = 20;                 % nm
params.cfbg.reflectivity = 0.20;            % peak power reflectivity
params.cfbg.dispersion = 0.1;               % ps/nm group delay slope

% Coupler ratios
params.couplers.rho = 0.45;                 % NALM 45/55 coupler
params.couplers.rho_out = 0.25;             % output coupler

% Numerical grid
params.grid.nt = 2^12;                      % number of temporal samples
params.grid.time_window = 40;               % total time window (ps)
params.grid.dz = 5e-6;                      % initial step size (km)
params.grid.tol = 1e-4;                     % adaptive tolerance

end
