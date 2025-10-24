function grid = initializeGrid(params)
%INITIALIZEGRID Prepare temporal and spectral grids for the simulation.

    Nt = params.Nt;
    T = params.time_window_s;
    dt = T / Nt;
    t = (-Nt/2:Nt/2-1) * dt;

    df = 1 / T;
    f = (-Nt/2:Nt/2-1) * df;
    omega = 2*pi*f;

    grid.t = t;
    grid.dt = dt;
    grid.f = fftshift(f);
    grid.df = df;
    grid.omega = fftshift(omega);
    grid.Nt = Nt;
    grid.time_window = T;
end
