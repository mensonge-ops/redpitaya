%DEMONALMSIMULATION Example script for the Yb401-PM NALM laser model.
%
%   This script configures the default cavity described in the user request
%   and runs the time-domain simulation.  The simulation may take several
%   minutes depending on the MATLAB/Octave version and hardware.
%
%   The results struct returned by nalm.runSimulation contains the final
%   field, per-round-trip spectra, pulse shapes, and inversion trajectory.

params = nalm.defaultParameters();
params.round_trips = 200;        % shorten runtime for a quick test
params.enable_plots = true;      % live diagnostics

results = nalm.runSimulation(params);

% Display summary of the final round trip
final_energy_nJ = results.output_energy(end) * 1e9;
final_inversion = results.inversion_ratio(end);

fprintf('Final output energy: %.3f nJ\n', final_energy_nJ);
fprintf('Inversion fraction: %.3f\n', final_inversion);
