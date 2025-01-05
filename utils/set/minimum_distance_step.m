function [dist2_new, z_opt_new, y_opt_new, out] = ...
    minimum_distance_step(dt, x1, dx1, C1, x2, dx2, C2, z_opt, y_opt, alg)
    % Performs a single integration step.
    % Inputs:
    %   dt: timestep for integration.
    %   x1, x2: Current parameters for C1 and C2.
    %   dx1, dx2: Rate of change of parameters x1 and x2.
    %   C1, C2: Convex sets, at least one of them strictly convex.
    %   z_opt: Optimal primal solution for x=(x1, x2).
    %   y_opt: Optimal dual solution for x=(x1, x2).
    %   alg: either 'euler' (default) or 'ode23t'.
    % Outputs:
    %   dist2_new: dist2 after dt time.
    %   z_opt_new: z_opt after dt time.
    %   y_opt_new: y_opt after dt time.
    MAX_KKT_ERR = 10;

    nz = C1.nz; % = C2.nz;

    if nargin < 10
        alg = 'euler';
    end

    if strcmp(alg, 'euler')
        [dz_opt, dy_opt, out] = ...
            minimum_distance_ode(x1, dx1, C1, x2, dx2, C2, z_opt, y_opt);
        z_opt_new = z_opt + dt * dz_opt;
        y_opt_new = y_opt + dt * dy_opt;
    elseif strcmp(alg, 'ode23t')
        % Stiff ODE solvers: ode15s, ode23s, ode23t, and ode23tb.
        % ode23t performs best.
        [~, zy_new] = ode23t(@(t,w) zy_ode(x1, dx1, C1, x2, dx2, C2, w), ...
            [0, dt], [z_opt; y_opt]);
        z_opt_new = zy_new(end, 1:2*nz)';
        y_opt_new = zy_new(end, 1+2*nz:end)';
    else
        error('alg must be either ''euler'' or ''ode23t''');
    end

    % KKT error at (x + dt * dx, z_opt_new, y_opt_new).
    z_opt_new1 = z_opt_new(1:nz);
    z_opt_new2 = z_opt_new(1+nz:end);
    y_opt_new1 = y_opt_new(1:C1.nr);
    y_opt_new2 = y_opt_new(1+C1.nr:end);
    [A1, ~, dAdz1, ~,~] = C1.derivatives(x1 + dt * dx1, z_opt_new1, y_opt_new1);
    [A2, ~, dAdz2, ~,~] = C2.derivatives(x2 + dt * dx2, z_opt_new2, y_opt_new2);
    kkt_new = [2 * (z_opt_new1 - z_opt_new2) + dAdz1' * y_opt_new1;
        2 * (z_opt_new2 - z_opt_new1) + dAdz2' * y_opt_new2; % Stationarity
        max([A1; A2], 0); % Primal feasibility
        min(y_opt_new, 0); % Dual feasibility
        y_opt_new' * [A1; A2] % Complementary slackness
        ];
    out.kkt_err = norm(kkt_new, 2);

    dist2_new = sum((z_opt_new1 - z_opt_new2).^2);

    out.opt_solution = 0;
    % If KKT error is large, reinitialize solution.
    if out.kkt_err > MAX_KKT_ERR
        [dist2_new, z_opt_new, y_opt_new ,~,~] = ...
            minimum_distance(x1 + dt * dx1, C1, x2 + dt * dx2, C2, z_opt, ...
            'interior-point');
        out.kkt_err = 0;
        out.opt_solution = 1;
    end
end


%% Function for minimum distance ODE.
function dzy_opt = zy_ode(x1, dx1, C1, x2, dx2, C2, zy_opt)
    nz = C1.nz; % = C2.nz
    z_opt = zy_opt(1:2*nz);
    y_opt = zy_opt(1+2*nz:end);
    [dz_opt, dy_opt, ~] = ...
        minimum_distance_ode(x1, dx1, C1, x2, dx2, C2, z_opt, y_opt);
    dzy_opt = [dz_opt; dy_opt];
end
