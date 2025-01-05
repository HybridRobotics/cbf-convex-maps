function [dist2, z_opt, y_opt, J, out] = ...
    minimum_distance(x1, C1, x2, C2, z0, alg)
    % Minimum distance between convex sets.
    %
    % Inputs:
    %   C1, C2: Convex sets, at least one of them strictly convex.
    %   x1, x2: Parameters for C1 and C2, respectively.
    %   z0: (optional) Warm start solution.
    %   alg: Optimization algorithm, either 'interior-point' (default) or 'sqp'.
    % Outputs:
    %   dist2: Square of minimum distance between C1 and C2.
    %   z_opt: Primal optimal solution (z1, z2).
    %   y_opt: Dual optimal solution (y1, y2).
    %   J: struct containing index set,
    %      = struct('J0c', J0c, 'J1', J1, 'J2e', J2e),
    %   out: = struct('run_time', run_time, 'status', status, 'iter', iter).
    EPS = 1e-6; % Margin for index set calculations.

    nz = C1.nz; % = C2.nz;

    if nargin < 6
        alg = 'interior-point';
    end
    assert(strcmp(alg, 'interior-point') || strcmp(alg, 'sqp'), ...
        'alg should be either ''interior-point'' or ''sqp''');
    if nargin < 5 || isempty(z0)
        z0 = zeros(2 * nz, 1);
    end

    % Objective function.
    obj  = @(z) objective(z, nz);
    cons = @(z) nonlcon(z, nz, x1, C1, x2, C2);

    % IPOpt settings.
    options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', alg);
    options.SpecifyConstraintGradient = true;
    options.SpecifyObjectiveGradient = true;
    options.ConstraintTolerance = 1e-8;
    options.OptimalityTolerance = 1e-7;
    if strcmp(alg, 'interior-point')
        options.HessianFcn = ...
            @(z, lambda) hessianfcn(z, lambda, nz, x1, C1, x2, C2);
        options.SubproblemAlgorithm = 'factorization';
    end
    options.CheckGradients = false;

    % Solve the minimum distance problem.
    dist_opt_start_time = tic;
    [z_opt, dist2, exitflag, output, lambda, ~,~] = ...
        fmincon(obj, z0, [],[],[],[],[],[], cons, options);
    out.run_time = toc(dist_opt_start_time);
    out.status = exitflag;
    out.iter = output.iterations;

    % Extract dual variables and index sets.
    y1 = lambda.ineqnonlin(1:C1.nr);
    y2 = lambda.ineqnonlin(C1.nr+1:end);
    y_opt = [y1; y2];

    z1 = z_opt(1:nz);
    z2 = z_opt(nz+1:end);
    % J0c: Inactive primal constraints at z_opt.
    % J1: Active (non-zero) dual variables.
    % J2e: Active primal constraints at z_opt, with zero dual.
    % Disjoint union of J0c, J1, and J2e is (1:C1.nr+C2.nr)'.
    J.J0c = find([C1.A(x1, z1); C2.A(x2, z2)] < -EPS);
    J.J1  = setdiff(find(y_opt > EPS), J.J0c);
    J.J2e = setdiff((1:C1.nr+C2.nr)', union(J.J0c, J.J1));
end

function [fun, grad] = objective(z, nz)
    z1 = z(1:nz);
    z2 = z(nz+1:end);
    fun = (z1 - z2)' * (z1 - z2);
    grad = 2*[z1 - z2; z2 - z1];
end

function [c, ceq, GC, GCeq] = nonlcon(z, nz, x1, C1, x2, C2)
    z1 = z(1:nz);
    z2 = z(nz+1:end);
    [A1, ~, dAdz1, ~,~] = C1.derivatives(x1, z1, zeros(C1.nr, 1));
    [A2, ~, dAdz2, ~,~] = C2.derivatives(x2, z2, zeros(C2.nr, 1));

    c = [A1; A2]';
    ceq = [];
    GC = sparse([dAdz1 zeros(C1.nr, nz); zeros(C2.nr, nz) dAdz2]');
    GCeq = [];
end

function H = hessianfcn(z, lambda, nz, x1, C1, x2, C2)
    z1 = z(1:nz);
    z2 = z(nz+1:end);
    y = lambda.ineqnonlin;
    y1 = y(1:C1.nr);
    y2 = y(C1.nr+1:end);
    [~,~,~,~, d2Adzz_y1] = C1.derivatives(x1, z1, y1);
    [~,~,~,~, d2Adzz_y2] = C2.derivatives(x2, z2, y2);

    H = [2 * eye(nz) + d2Adzz_y1, -2 * eye(nz);
        -2 * eye(nz), 2 * eye(nz) + d2Adzz_y2];
    H = sparse(H);
end
