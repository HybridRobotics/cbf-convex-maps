clear

current_path = path;
path(path, genpath('..\'));


%% Define convex sets.
V = [-1, -1; -1, 1; 1, 1; 1, -1; 0, 0; 1, 1; 0, 2];
V = [V, -ones(7, 1); V, ones(7, 1)];
Q = [3, 1, 0; 1, 2, 0; 0, 0, 1];
sp = StaticPolytope(V);
smp = SmoothPolytope(3, sp, 10);
re = RigidEllipsoid(3, Q);


%% Define state sequences.
dt = 0.01;
T = 25;
t_seq = 0:dt:T;
nT = length(t_seq);

x_sp_seq = zeros(0, nT);
p_smp_seq = [10 * cos(2 * pi * t_seq / 10);
    10 * sin(2 * pi * t_seq / 10);
    3 * sin(2 * pi * t_seq / 11)];
rot_z_seq = 2 * pi * t_seq / 9;
R_smp_seq = [cos(rot_z_seq); sin(rot_z_seq); zeros(1, nT); -sin(rot_z_seq); ...
    cos(rot_z_seq); zeros(3, nT); ones(1, nT)];
x_smp_seq = [p_smp_seq; R_smp_seq];
x_re_seq = x_smp_seq;


%% Minimum distance ODE test between StaticPolytope and SmoothPolytope.

out_smp = compare_opt_ode(dt, t_seq, x_sp_seq, sp, x_smp_seq, smp);

figure();
hold on
plot(t_seq, sqrt(out_smp.dist2_opt_seq), '-b');
plot(t_seq, sqrt(out_smp.dist2_ode_seq), '-r');
hold off;
disp(['Avg. opt solution time (s): ' num2str(mean(out_smp.t_opt_seq))]);
disp(['Avg. ODE solution time (s): ' num2str(mean(out_smp.t_ode_seq))]);
dist_err = norm(sqrt(out_smp.dist2_opt_seq) - sqrt(out_smp.dist2_ode_seq), Inf);
zopt_err = norm(vecnorm(out_smp.zopt_opt_seq - out_smp.zopt_ode_seq, 2), Inf);
yopt_err_seq = vecnorm(out_smp.yopt_opt_seq - out_smp.yopt_ode_seq, 2);
yopt_rel_err_seq = yopt_err_seq / vecnorm(out_smp.yopt_opt_seq, 2);
yopt_rel_err = norm(yopt_rel_err_seq, Inf);
disp(['Distance Inf norm error (m): ' num2str(dist_err)]);
disp(['Primal solution Inf norm error (m): ' num2str(zopt_err)]);
disp(['Dual solution Inf norm relative error (): ' num2str(yopt_rel_err)]);
disp(['Number of optimization solutions: ' num2str(out_smp.n_opt_solution)]);

figure();
hold on;
plot(t_seq, out_smp.kkt_err_seq, '-b');
plot(t_seq, sqrt(out_smp.dist2_opt_seq) - sqrt(out_smp.dist2_ode_seq), '-r');
hold off;


%% Minimum distance ODE test between StaticPolytope and RigidEllipsoid.

out_re = compare_opt_ode(dt, t_seq, x_sp_seq, sp, x_re_seq, re);

figure();
hold on
plot(t_seq, sqrt(out_re.dist2_opt_seq), '-b');
plot(t_seq, sqrt(out_re.dist2_ode_seq), '-r');
hold off;
disp(['Avg. opt solution time (s): ' num2str(mean(out_re.t_opt_seq))]);
disp(['Avg. ODE solution time (s): ' num2str(mean(out_re.t_ode_seq))]);
dist_err = norm(sqrt(out_re.dist2_opt_seq) - sqrt(out_re.dist2_ode_seq), Inf);
zopt_err = norm(vecnorm(out_re.zopt_opt_seq - out_re.zopt_ode_seq, 2), Inf);
yopt_err_seq = vecnorm(out_re.yopt_opt_seq - out_re.yopt_ode_seq, 2);
yopt_rel_err_seq = yopt_err_seq / vecnorm(out_re.yopt_opt_seq, 2);
yopt_rel_err = norm(yopt_rel_err_seq, Inf);
disp(['Distance Inf norm error (m): ' num2str(dist_err)]);
disp(['Primal solution Inf norm error (m): ' num2str(zopt_err)]);
disp(['Dual solution Inf norm relative error (): ' num2str(yopt_rel_err)]);
disp(['Number of optimization solutions: ' num2str(out_re.n_opt_solution)]);

figure();
hold on;
plot(t_seq, out_re.kkt_err_seq, '-b');
plot(t_seq, sqrt(out_re.dist2_opt_seq) - sqrt(out_re.dist2_ode_seq), '-r');
hold off;

%%
path(current_path);


%% Function to compare solutions from optimization and ODE.
function [out] = compare_opt_ode(dt, t_seq, x_seq1, C1, x_seq2, C2)
    nT = length(t_seq);
    nz = C1.nz; % = C2.nz

    dist2_opt_seq = zeros(1, nT);
    dist2_ode_seq = zeros(1, nT);
    zopt_opt_seq = zeros(2 * nz, nT);
    zopt_ode_seq = zeros(2 * nz, nT);
    yopt_opt_seq = zeros(C1.nr + C2.nr, nT);
    yopt_ode_seq = zeros(C1.nr + C2.nr, nT);
    t_opt_seq = zeros(1, nT);
    t_ode_seq = zeros(1, nT);
    kkt_err_seq = zeros(1, nT);

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
    n_J2e = 0;
    n_opt_solution = 0;

    wb = waitbar(0, 'Starting distance ODE');
    for k = 1:nT-1
        x1 = x_seq1(:, k);
        x2 = x_seq2(:, k);
        dx1 = (x_seq1(:, k+1) - x_seq1(:, k)) / dt;
        dx2 = (x_seq2(:, k+1) - x_seq2(:, k)) / dt;
        z_opt = zopt_ode_seq(:, k);
        y_opt = yopt_ode_seq(:, k);

        tic_ = tic;
        [dist2_ode_seq(k+1), zopt_ode_seq(:, k+1), yopt_ode_seq(:, k+1), o_] = ...
            minimum_distance_step(dt, x1, dx1, C1, x2, dx2, C2, ...
            z_opt, y_opt, 'euler');
        t_ode_seq(k) = toc(tic_);
        if isfield(o_, 'J2e')
            n_J2e = n_J2e + o_.J2e;
        end
        kkt_err_seq(k) = o_.kkt_err;
        n_opt_solution = n_opt_solution + o_.opt_solution;
        waitbar(k/nT, wb, sprintf('Progress: %d %%', floor(k/nT*100)));
    end
    close(wb);

    out.dist2_opt_seq = dist2_opt_seq;
    out.dist2_ode_seq = dist2_ode_seq;
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
