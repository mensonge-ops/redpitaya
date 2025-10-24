function updatePlots(figs, grid, metrics, rt)
%UPDATEPLOTS Update live figures with the latest metrics.

    if isempty(figs)
        return;
    end

    set(figs.time_line, 'YData', abs(metrics.output_pulse).^2);

    freq_axis = grid.f / 1e12;
    set(figs.spec_line, 'XData', freq_axis, 'YData', metrics.output_spectrum);

    if rt == 1
        figs.inv_line.XData = rt;
        figs.inv_line.YData = metrics.inversion_ratio;
        figs.energy_line.XData = rt;
        figs.energy_line.YData = metrics.output_energy * 1e9;
    else
        figs.inv_line.XData(end+1) = rt;
        figs.inv_line.YData(end+1) = metrics.inversion_ratio;
        figs.energy_line.XData(end+1) = rt;
        figs.energy_line.YData(end+1) = metrics.output_energy * 1e9;
    end

    drawnow limitrate;
end
