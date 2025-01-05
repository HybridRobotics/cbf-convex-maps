%% Function to compare solutions from optimization and ODE.
function [out] = compare_opt_ode(t_seq, x_seq1, C1, x_seq2, C2)
    dt = t_seq(2) - t_seq(1);
    nT = length(t_seq);
    nz = C1.nz; % = C2.nz

    dist2_opt_seq = zeros(1, nT); % Square of minimum distance.
    dist2_ode_seq = zeros(1, nT);
    zopt_opt_seq = zeros(2 * nz, nT); % Primal optimal solution.
    zopt_ode_seq = zeros(2 * nz, nT);
    yopt_opt_seq = zeros(C1.nr + C2.nr, nT); % Dual optimal solution.
    yopt_ode_seq = zeros(C1.nr + C2.nr, nT);
    t_opt_seq = zeros(1, nT); % Solution time.
    t_ode_seq = zeros(1, nT);
    kkt_err_seq = zeros(1, nT); % 2-norm of KKT residual.

    % Compute minimum distance using optimization.
    wb = waitbar(0, 'Starting distance optimization');
    for k = 1:nT
        x1 = x_seq1(:, k);
        x2 = x_seq2(:, k);
        if k == 1
            ws = zeros(2*nz, 1);
        else
            ws = zopt_opt_seq(:, k-1);
        end
        tic_ = tic;
        [dist2_opt_seq(k), zopt_opt_seq(:, k), yopt_opt_seq(:, k) ,~,~] = ...
            minimum_distance(x1, C1, x2, C2, ws, 'interior-point');
        t_opt_seq(k) = toc(tic_);
        waitbar(k/nT, wb, sprintf('Progress: %d %%', floor(k/nT*100)));
    end
    close(wb);

    % Compute minimum distance using ODE.
    dist2_ode_seq(1) = dist2_opt_seq(1);
    zopt_ode_seq(:, 1) = zopt_opt_seq(:, 1);
    yopt_ode_seq(:, 1) = yopt_opt_seq(:, 1);
    n_J2e = 0; % Number of times J_2e is non-empty.
    n_opt_solution = 0; % Number of times solution is reinitialized.

    wb = waitbar(0, 'Starting distance ODE');
    for k = 1:nT-1
        x1 = x_seq1(:, k);
        x2 = x_seq2(:, k);
        dx1 = (x_seq1(:, k+1) - x_seq1(:, k)) / dt;
        dx2 = (x_seq2(:, k+1) - x_seq2(:, k)) / dt;
        z_opt = zopt_ode_seq(:, k);
        y_opt = yopt_ode_seq(:, k);

        tic_ = tic;
        [dist2_ode_seq(k+1), zopt_ode_seq(:, k+1), yopt_ode_seq(:, k+1), out_] = ...
            minimum_distance_step(dt, x1, dx1, C1, x2, dx2, C2, ...
            z_opt, y_opt, 'euler');
        t_ode_seq(k) = toc(tic_);
        if isfield(out_, 'J2e')
            n_J2e = n_J2e + out_.J2e;
        end
        kkt_err_seq(k) = out_.kkt_err;
        n_opt_solution = n_opt_solution + out_.opt_solution;
        waitbar(k/nT, wb, sprintf('Progress: %d %%', floor(k/nT*100)));
    end
    close(wb);

    % Distance derivative.
    Ddist2_opt_seq = gradient(dist2_opt_seq(1:end-1), dt); % diff(dist2_opt_seq) / dt;
    Ddist2_ode_seq = gradient(dist2_ode_seq(1:end-1), dt); % diff(dist2_ode_seq) / dt;
    Ddist2_analytic_seq = zeros(size(Ddist2_ode_seq));
    for k = 1:nT-1
        x1 = x_seq1(:, k);
        x2 = x_seq2(:, k);
        if k == 1
            dx1 = (x_seq1(:, 2) - x_seq1(:, 1)) / dt;
            dx2 = (x_seq2(:, 2) - x_seq2(:, 1)) / dt;
        else
            dx1 = (x_seq1(:, k+1) - x_seq1(:, k-1)) / (2*dt);
            dx2 = (x_seq2(:, k+1) - x_seq2(:, k-1)) / (2*dt);
        end
        z1 = zopt_ode_seq(1:nz, k);
        z2 = zopt_ode_seq(nz+1:end, k);
        y1 = yopt_ode_seq(1:C1.nr, k);
        y2 = yopt_ode_seq(C1.nr+1:end, k);

        [~, dAdx1, ~,~,~] = C1.derivatives(x1, z1, zeros(C1.nr, 1));
        [~, dAdx2, ~,~,~] = C2.derivatives(x2, z2, zeros(C2.nr, 1));
        Ddist2_analytic_seq(k) = y1' * dAdx1 * dx1 + y2' * dAdx2 * dx2;
    end

    out.t_seq = t_seq;
    out.dist2_opt_seq = dist2_opt_seq;
    out.dist2_ode_seq = dist2_ode_seq;
    out.Ddist2_opt_seq = Ddist2_opt_seq;
    out.Ddist2_ode_seq = Ddist2_ode_seq;
    out.Ddist2_analytic_seq = Ddist2_analytic_seq;
    out.zopt_opt_seq = zopt_opt_seq;
    out.zopt_ode_seq = zopt_ode_seq;
    out.yopt_opt_seq = yopt_opt_seq;
    out.yopt_ode_seq = yopt_ode_seq;
    out.t_opt_seq = t_opt_seq;
    out.t_ode_seq = t_ode_seq;
    out.n_J2e = n_J2e;
    out.kkt_err_seq = kkt_err_seq;
    out.n_opt_solution = n_opt_solution;
end
