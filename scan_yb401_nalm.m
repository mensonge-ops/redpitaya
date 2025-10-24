function scan_results = scan_yb401_nalm(base_params, base_options, sweep)
%SCAN_YB401_NALM  Coarse parameter scan for the Yb401-PM NALM cavity.
%   scan_results = SCAN_YB401_NALM(base_params, base_options, sweep)
%   iterates over the provided parameter grids (coupler ratios, gain levels,
%   initial pulse widths) and records the resulting bandwidth and lock status.
%   It is intended to help locate parameter combinations that produce >20 nm
%   spectra or stable soliton-like pulses.
%
%   The SWEEP struct may define the following vectors:
%       rho_values           - NALM coupler ratios
%       rho_out_values       - output coupler ratios
%       gain_main_values     - main gain gssdB values
%       gain_nalm_values     - NALM gain gssdB values
%       tfwhm_values         - input pulse widths (ps)
%       N2_values            - soliton order squared values
%       max_round_trips      - override for per-simulation trip count
%
%   Results are returned in scan_results.results (struct array) and the best
%   candidate (largest spectral width that satisfies the lock criterion) is
%   returned as scan_results.best.  Each entry includes the final metrics,
%   lock status, and updated parameter set.

if nargin < 1 || isempty(base_params)
    base_params = default_yb401_params();
end
if nargin < 2
    base_options = struct();
end
if nargin < 3
    sweep = struct();
end

% Default sweeps
if ~isfield(sweep, 'rho_values') || isempty(sweep.rho_values)
    sweep.rho_values = 0.4:0.05:0.55;
end
if ~isfield(sweep, 'rho_out_values') || isempty(sweep.rho_out_values)
    sweep.rho_out_values = 0.2:0.05:0.3;
end
if ~isfield(sweep, 'gain_main_values') || isempty(sweep.gain_main_values)
    sweep.gain_main_values = 34:2:40;
end
if ~isfield(sweep, 'gain_nalm_values') || isempty(sweep.gain_nalm_values)
    sweep.gain_nalm_values = 24:2:30;
end
if ~isfield(sweep, 'tfwhm_values') || isempty(sweep.tfwhm_values)
    sweep.tfwhm_values = [3.0 3.5 4.0];
end
if ~isfield(sweep, 'N2_values') || isempty(sweep.N2_values)
    sweep.N2_values = [1.2^2 1.35^2 1.5^2];
end
if ~isfield(sweep, 'max_round_trips') || isempty(sweep.max_round_trips)
    sweep.max_round_trips = 80;
end

local_opt = base_options;
local_opt.collect_history = false;
local_opt.collect_final_traces = false;
local_opt.live_monitor = false;
local_opt.verbose = false;
local_opt.max_round_trips = sweep.max_round_trips;

idx = 0;
results = struct([]);

for rho = sweep.rho_values
    for rho_out = sweep.rho_out_values
        for g_main = sweep.gain_main_values
            for g_nalm = sweep.gain_nalm_values
                for tfwhm = sweep.tfwhm_values
                    for N2 = sweep.N2_values
                        idx = idx + 1;
                        params = base_params;
                        params.couplers.rho = rho;
                        params.couplers.rho_out = rho_out;
                        params.sections.amf_main.gssdB = g_main;
                        params.sections.amf_nalm.gssdB = g_nalm;
                        params.pulse.tfwhm = tfwhm;
                        params.pulse.N2 = N2;

                        [sim_result, ~] = run_yb401_nalm(params, local_opt);
                        final_metrics = sim_result.final_metrics;

                        results(idx).params = params; %#ok<AGROW>
                        results(idx).result = sim_result; %#ok<AGROW>
                        results(idx).spectral_width_nm = final_metrics.spectral_width_nm; %#ok<AGROW>
                        results(idx).locked = sim_result.locked; %#ok<AGROW>
                        results(idx).round_trips = sim_result.round_trips_executed; %#ok<AGROW>
                    end
                end
            end
        end
    end
end

if isempty(results)
    scan_results = struct('results', [], 'best', []);
    return;
end

% Identify best locked candidate; if none locked, pick widest spectrum.
locked_idx = find([results.locked]);
if ~isempty(locked_idx)
    [~, rel_idx] = max([results(locked_idx).spectral_width_nm]);
    best_idx = locked_idx(rel_idx);
else
    [~, best_idx] = max([results.spectral_width_nm]);
end

scan_results = struct();
scan_results.results = results;
scan_results.best = results(best_idx);

end
