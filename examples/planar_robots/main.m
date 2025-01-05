%% Example 2: Planar robots.
% Example demonstrating convex set CBFs for planar robots.
clear
clear control

path_ = path;
path(path_, genpath('..\..\'));


%% Parameters and hyper-parameters.
ALPHA_CBF = 1;
ALPHA_CLF = 1;
WEIGHT_CLF = 5; % relative cost of CLF violation compared to input cost.
DIST_MARGIN = 0.1; % [m^2].

DT_CONTROL = 0.01; % [s].
T_SIM = 100; % [s].
REL_FREQ_SIM = 5; % [], Relative frequency of simulation compared to control.


%% Initialize environment.
t_seq = 0:DT_CONTROL:T_SIM;
nT = length(t_seq);

[robots, xf, clfs] = set_env();
nrobots = numel(robots);
[dist_struct, logs] = init(robots, t_seq, xf);


%% Run simulation.
wb = waitbar(0, 'Starting simulation');
for t = 1:nT-1
    % Compute control input U.
    [U, dist_struct, solve_time] = control(robots, dist_struct, xf, clfs, ...
    DT_CONTROL, ALPHA_CBF, ALPHA_CLF, WEIGHT_CLF, DIST_MARGIN);
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
            logs.dist2_seq{i, j}(t+1) = dist_struct{i, j}.dist2;
            logs.zopt_seq{i, j}(:, t+1) = dist_struct{i, j}.zopt;
            logs.yopt_seq{i, j}(:, t+1) = dist_struct{i, j}.yopt;
        end
    end
    waitbar(t/(nT-1), wb, sprintf('time: %f s', t_seq(t)));
end
close(wb);


%% Statistics.
dist_solve_time = zeros(nrobots, nT - 1);
qp_solve_time = zeros(nrobots, nT - 1);
for i = 1:nrobots
    dist_solve_time(i, :) = logs.solve_time_seq{i}(1, :);
    qp_solve_time(i, :) = logs.solve_time_seq{i}(2, :);
end
disp(['avg dist solve time =', num2str(mean(dist_solve_time, 'all'))]);
disp(['avg QP solve time =', num2str(mean(qp_solve_time, 'all'))]);


%% Plot graphs.
% Trajectory + snapshots.
figure()
ax = gca;
for i = 1:nrobots
    hold on
    plot(ax, logs.x_seq{i}(1,:), logs.x_seq{i}(2,:), '--');
end

% plot(t_seq, logs.x_seq{7}(2,:), '--');

% Distance plots.
figure()
ax = gca;
dist2_seq_ = Inf * ones(nrobots, nrobots, nT);
for i = 1:nrobots
    for j = 1:nrobots
        if i == j
            continue
        end
        dist2_seq_(i, j, :) = logs.dist2_seq{i, j};
    end
end
min_dist2_seq = squeeze(min(dist2_seq_, [], [1, 2]));
for i = 1:nrobots
    if i == 1
        continue
    end
    hold on
    plot(ax, t_seq, logs.dist2_seq{1, i}, '--k');
end
plot(ax, t_seq, min_dist2_seq, '-r');
ylim([10^-3 1000]);
set(gca, 'YScale', 'log');
% Add dist margin line.
