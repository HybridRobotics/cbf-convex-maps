function [dist_struct, logs] = init(robots, t_seq, xf)
    nrobots = numel(robots);

    % Initialize distances.
    dist_struct = cell(nrobots, nrobots);
    for i = 1:nrobots
        for j = 1:nrobots
            if i == j
                continue;
            end
            x1 = robots{i}.system.x;
            x2 = robots{j}.system.x;
            C1 = robots{i}.sets{1};
            C2 = robots{j}.sets{1};
            s = struct();
            s.nr1 = C1.nr;
            s.nr2 = C2.nr;
            [s.dist2, s.zopt, s.yopt, ~,~] = minimum_distance(x1, C1, x2, C2, ...
                [], 'interior-point');
            dist_struct{i, j} = s;
        end
    end
    disp('Initialized pairwise distances');

    % Set up logs.
    logs.t_seq = t_seq;
    logs.dt = t_seq(2) - t_seq(1);
    logs.Ti = t_seq(1);
    logs.Tf = t_seq(end);
    logs.nT = length(t_seq);
    logs.nrobots = numel(robots);

    logs.xf = xf;

    logs.x_seq = cell(nrobots, 1);
    logs.u_seq = cell(nrobots, 1);
    logs.dist2_seq = cell(nrobots, nrobots);
    logs.zopt_seq = cell(nrobots, nrobots);
    logs.yopt_seq = cell(nrobots, nrobots);
    logs.solve_time_seq = cell(nrobots, 1);
    logs.out = cell(nrobots, nrobots);
    for i = 1:nrobots
        logs.x_seq{i} = zeros(robots{i}.system.nx, logs.nT);
        logs.u_seq{i} = zeros(robots{i}.system.nu, logs.nT - 1);
        logs.solve_time_seq{i} = zeros(2, logs.nT - 1); % (dist, cbf).

        logs.x_seq{i}(:, 1) = robots{i}.system.x;
    end
    for i = 1:nrobots
        for j = 1:nrobots
            if i == j
                continue
            end
            logs.dist2_seq{i, j} = zeros(1, logs.nT);
            logs.zopt_seq{i, j} = zeros(8, logs.nT);
            logs.yopt_seq{i, j} = ...
                zeros(dist_struct{i, j}.nr1 + dist_struct{i, j}.nr2, logs.nT);
            logs.out{i, j} = zeros(2, logs.nT); % (kkt_err, opt_solution).

            logs.dist2_seq{i, j}(1) = dist_struct{i, j}.dist2;
            logs.zopt_seq{i, j}(:, 1) = dist_struct{i, j}.zopt;
            logs.yopt_seq{i, j}(:, 1) = dist_struct{i, j}.yopt;
            logs.out{i, j}(2, 1) = 1;
        end
    end
    disp('Initialized logs');
end
