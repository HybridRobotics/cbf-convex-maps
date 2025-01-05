%% Example 3: 3D robots.
% Example demonstrating convex set CBFs for 3D robots.

clear
clear control

path_ = path;
path(path_, genpath('..\..\'));


%% Parameters and hyper-parameters.
ALPHA_HCBF = 1;
ALPHA_VCBF = 1/5;
ALPHA_CLF = 1;
WEIGHT_CLF = 100; % relative cost of CLF violation compared to input cost.
DIST_MARGIN = 0.1; % [m^2].
VMAX = 0.5; % [m/s].

DT_CONTROL = 0.01; % [s].
T_SIM = 70; % [s].
REL_FREQ_SIM = 5; % [], Relative frequency of simulation compared to control.


%% Initialize environment.
t_seq = 0:DT_CONTROL:T_SIM;
nT = length(t_seq);

if ~isfile("init_vars.mat")
    [robots, xf, tracking_clf] = set_env();
    save("init_vars.mat", "robots", "tracking_clf", "xf", "-mat");
else
    load("init_vars.mat");
    figure();
    ax = gca;
    for i = 1:numel(robots)
        hold on
        if 2*i <= numel(robots)
            c_ = 'r';
        else
            c_ = 'g';
        end
        robots{i}.sets{1}.plot_surf(robots{i}.system.x, ax, c_, 1, []);
    end
    axis equal
end
nrobots = numel(robots);
[dist_struct, logs] = init(robots, t_seq, xf);

%
init_dist = zeros(nrobots, nrobots);
for i = 1:nrobots
    for j = 1:nrobots
        if i == j
            continue;
        end
        init_dist(i, j) = dist_struct{i, j}.dist2;
    end
end


%% Run simulation.
wb = waitbar(0, 'Starting simulation');
for t = 1:nT-1
    % Compute control input U.
    [U, dist_struct, solve_time, out] = control(robots, dist_struct, xf, ...
        tracking_clf, VMAX, DT_CONTROL, ALPHA_HCBF, ALPHA_VCBF, ...
        ALPHA_CLF, WEIGHT_CLF, DIST_MARGIN);
    % Update states.
    for i = 1:REL_FREQ_SIM
        for j = 1:nrobots
            robots{j}.system.x = ...
                robots{j}.system.step(DT_CONTROL / REL_FREQ_SIM, U{j});
        end
    end
    % Update logs.
    for i = 1:nrobots
        logs.x_seq{i}(:, t+1) = robots{i}.system.x;
        logs.u_seq{i}(:, t) = U{i};
        logs.solve_time_seq{i}(:, t) = solve_time{i};
        for j = 1:nrobots
            if i == j
                continue;
            end
            logs.dist2_seq{i, j}(t+1) = dist_struct{i, j}.dist2;
            logs.zopt_seq{i, j}(:, t+1) = dist_struct{i, j}.zopt;
            logs.yopt_seq{i, j}(:, t+1) = dist_struct{i, j}.yopt;
            logs.out{i, j}(1, t+1) = out{i, j}.kkt_err;
            logs.out{i, j}(2, t+1) = out{i, j}.opt_solution;
        end
    end
    waitbar(t/(nT-1), wb, sprintf('time: %f s', t_seq(t)));
end
close(wb);

%% Solve for the actual distance.
logs.opt_dist2_seq = cell(nrobots, nrobots);
logs.opt_zopt_seq = cell(nrobots, nrobots);
logs.opt_yopt_seq = cell(nrobots, nrobots);
iter = 0;
total_iter = nrobots * (nrobots - 1) / 2 * nT;
wb = waitbar(0, 'Starting distance computation');
for i = 1:nrobots
    for j = i+1:nrobots
        C1 = robots{i}.sets{1};
        C2 = robots{j}.sets{1};
        logs.opt_dist2_seq{i, j} = zeros(1, nT);
        logs.opt_zopt_seq{i, j} = zeros(8, nT);
        logs.opt_yopt_seq{i, j} = zeros(C1.nr + C2.nr, nT);
        for t = 1:nT
            x1 = logs.x_seq{i}(:, t);
            x2 = logs.x_seq{j}(:, t);
            if t == 1
                ws = zeros(8, 1);
            else
                ws = logs.opt_zopt_seq{i, j}(:, t-1);
            end
            [dist2, zopt, yopt, ~,~] = minimum_distance(x1, C1, x2, C2, ...
                    ws, 'interior-point');
            logs.opt_dist2_seq{i, j}(t) = dist2;
            logs.opt_zopt_seq{i, j}(:, t) = zopt;
            logs.opt_yopt_seq{i, j}(:, t) = yopt;
            iter = iter + 1;
            waitbar(iter/total_iter, wb, ...
                sprintf('Completed: %.2f%%', iter/total_iter * 100));
        end
    end
end
close(wb);

%% Solve for distance derivative and KKT error.
dt = logs.dt;
dist2_ode_seq = zeros(nrobots * (nrobots - 1) / 2, nT);
dist2_opt_seq = zeros(nrobots * (nrobots - 1) / 2, nT);
idx_ = 1;
for i = 1:nrobots
    for j = i+1:nrobots
        dist2_ode_seq(idx_, :) = logs.dist2_seq{i, j};
        dist2_opt_seq(idx_, :) = logs.opt_dist2_seq{i, j};
        idx_ = idx_ + 1;
    end
end
Ddist2_ode_seq = zeros(nrobots * (nrobots - 1) / 2, nT - 1);
Ddist2_opt_seq = zeros(nrobots * (nrobots - 1) / 2, nT - 1);
kkt_err_seq = zeros(nrobots * (nrobots - 1) / 2, nT - 1);
idx_ = 1;
total_iter = nrobots * (nrobots - 1) / 2;
wb = waitbar(0, 'Starting distance derivative computation');
for i = 1:nrobots
    for j = i+1:nrobots
        Ddist2_opt_seq(idx_, :) = gradient(dist2_opt_seq(idx_, 1:end-1), dt);
        C1 = robots{i}.sets{1};
        C2 = robots{j}.sets{1};
        for k = 1:nT-1
            x1 = logs.x_seq{i}(:, k);
            x2 = logs.x_seq{j}(:, k);
            if k == 1
                dx1 = (logs.x_seq{i}(:, 2) - x1) / dt;
                dx2 = (logs.x_seq{j}(:, 2) - x2) / dt;
            else
                dx1 = (logs.x_seq{i}(:, k+1) - logs.x_seq{i}(:, k-1)) / (2*dt);
                dx2 = (logs.x_seq{j}(:, k+1) - logs.x_seq{j}(:, k-1)) / (2*dt);
            end
            z1 = logs.zopt_seq{i, j}(1:4, k);
            z2 = logs.zopt_seq{i, j}(5:8, k);
            y1 = logs.yopt_seq{i, j}(1, k);
            y2 = logs.yopt_seq{i, j}(2, k);

            [A1, dAdx1, dAdz1, ~,~] = C1.derivatives(x1, z1, zeros(C1.nr, 1));
            [A2, dAdx2, dAdz2, ~,~] = C2.derivatives(x2, z2, zeros(C2.nr, 1));
            Ddist2_ode_seq(idx_, k) = y1' * dAdx1 * dx1 + y2' * dAdx2 * dx2;

            stationarity = [2 * (z1 - z2) + dAdz1' * y1;
                2 * (z2 - z1) + dAdz2' * y2];
            primal_feasibility = [max(A1, 0); max(A2, 0)];
            dual_feasibility = [min(y1, 0); min(y2, 0)];
            complementary_slackness = [y1' * A1; y2' * A2];
            kkt_err_seq(idx_, k) = norm([stationarity; primal_feasibility;
                dual_feasibility; complementary_slackness], 2);
        end
        idx_ = idx_ + 1;
        waitbar(idx_/total_iter, wb, ...
                sprintf('Completed: %.2f%%', idx_/total_iter * 100));
    end
end
close(wb);
logs.Ddist2_seq = Ddist2_ode_seq;
logs.opt_Ddist2_seq = Ddist2_opt_seq;
logs.kkt_err_seq = kkt_err_seq;


%% Statistics.
print_stats(logs);


%% Plot figure.
% NOTE: Change figure size to [370, 300] points.
til = plot_figure(robots, logs, ALPHA_HCBF, DIST_MARGIN);
save_plot_as_one = false;
if true
    if save_plot_as_one
        fig = gcf;
        print(fig, strcat('../../fig/3d-cbf.png'), '-dpng', '-r1000');
        print(fig, strcat('../../fig/3d-cbf.eps'), '-depsc', '-r600');
    else
        child_ = flipud(til.Children);
        child_ = [child_(1:2:3); child_(5:2:end)];
        title = '../../fig/3d-cbf-';
        subtitles = {'env-init', 'env-final', 'dist', 'kkt', 'rel-dist', 'Ddist'};
        for i = 1:6
            exportgraphics(child_(i), strcat(title, subtitles{i}, '.png'), ...
                Resolution = 1000);
            exportgraphics(child_(i), strcat(title, subtitles{i}, '.eps'), ...
                Resolution = 600);
        end
    end
end


%% Plot environment.
% NOTE: Change figure size to [370, _] points.
% til = plot_env(robots, logs);
% fig = gcf;
% print(fig, strcat('../../fig/3d-cbf-env.png'), '-dpng', '-r1000');
% print(fig, strcat('../../fig/3d-cbf-env.eps'), '-depsc', '-r600');
