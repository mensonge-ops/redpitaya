function data = LoadYbCrossSections()
%LOADYBCROSSSECTIONS Load tabulated Yb401-PM cross sections.
%   DATA = LOADYBCROSSSECTIONS() reads the CSV file containing the tabulated
%   absorption and emission cross sections for the Nufern Yb401-PM fiber.
%   The returned struct contains the fields:
%       wavelength_nm - column vector of wavelengths in nm
%       sigma_abs     - absorption cross section (m^2)
%       sigma_em      - emission cross section (m^2)
%
%   The tabulated data was digitised from the manufacturer application note
%   and fits the multi-Gaussian model that is commonly used for Yb-doped
%   silica fibers.  The table spans 900-1100 nm which covers the 976 nm pump
%   band and the 1030 nm signal band used in the NALM cavity.
%
%   See also GETYB401PMGAINPARAMS.
%
%   This function requires no toolboxes and relies only on MATLAB's CSV
%   parser.  If the table is missing, an informative error is raised.

    tablePath = fullfile(fileparts(mfilename('fullpath')), '..', 'data', ...
        'yb401pm_cross_sections.csv');

    if ~exist(tablePath, 'file')
        error('LoadYbCrossSections:MissingTable', ...
            'Cross section table not found at %s', tablePath);
    end

    raw = readmatrix(tablePath, 'NumHeaderLines', 1);
    if size(raw, 2) < 3
        error('LoadYbCrossSections:InvalidTable', ...
            'Unexpected number of columns in %s', tablePath);
    end

    data = struct();
    data.wavelength_nm = raw(:, 1);
    data.sigma_abs = raw(:, 2);
    data.sigma_em = raw(:, 3);
end
