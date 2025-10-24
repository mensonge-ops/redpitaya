function figs = setupPlots(grid, params)
%SETUPPLOTS Create live-plotting figures.

    figs.figure = figure('Name', 'NALM Simulation', 'NumberTitle', 'off');
    tiledlayout(figs.figure, 2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    % Time-domain pulse
    figs.ax_time = nexttile(figs.figure);
    figs.time_line = plot(figs.ax_time, grid.t * 1e12, zeros(size(grid.t)));
    xlabel(figs.ax_time, 'Time (ps)');
    ylabel(figs.ax_time, '|E|^2 (W)');
    title(figs.ax_time, 'Output Pulse');

    % Spectrum
    figs.ax_spec = nexttile(figs.figure);
    figs.spec_line = plot(figs.ax_spec, grid.f / 1e12, zeros(size(grid.f)));
    xlabel(figs.ax_spec, 'Frequency Offset (THz)');
    ylabel(figs.ax_spec, 'PSD (a.u.)');
    title(figs.ax_spec, 'Output Spectrum');

    % Inversion trace
    figs.ax_inv = nexttile(figs.figure);
    figs.inv_line = plot(figs.ax_inv, nan, nan);
    xlabel(figs.ax_inv, 'Round trip');
    ylabel(figs.ax_inv, 'N_2 / N_{tot}');
    title(figs.ax_inv, 'Inversion');

    % Energy trace
    figs.ax_energy = nexttile(figs.figure);
    figs.energy_line = plot(figs.ax_energy, nan, nan, '-o');
    xlabel(figs.ax_energy, 'Round trip');
    ylabel(figs.ax_energy, 'Energy (nJ)');
    title(figs.ax_energy, 'Output Energy');

    drawnow;
end
