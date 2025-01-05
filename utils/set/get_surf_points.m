function [V] = get_surf_points(C, x, n)
    % Find points on the surface of a 2d/3d convex set.
    %
    % Inputs:
    %   C: Convex set, C.radius and C.center should be defined.
    %   x: State of C.
    %   n: Number of points on the surface.
    % Outputs:
    %   V: [_, nz] matrix of surface points.

    if nargin < 3
        n = 25;
    end

    if C.nz == 3
        [x_, y_, z_] = sphere(n);
        x_ = x_(:); y_ = y_(:); z_ = z_(:);
        P = C.radius * [x_, y_, z_] + ones(length(x_), 1) * C.center';
    elseif C.nz == 2
        angles = (0:n-1)' / n * 2*pi;
        P = C.radius * [cos(angles), sin(angles)] + ...
            ones(length(n), 1) * C.center';
    else
        error('nz other than 2 or 3 is not supported');
    end

    V = zeros(size(P));

    options = optimoptions('fmincon', 'Display', 'off', ...
        'Algorithm', 'interior-point');
    options.SubproblemAlgorithm = 'factorization';
    options.SpecifyConstraintGradient = true;
    options.SpecifyObjectiveGradient  = true;
    options.ConstraintTolerance = 1e-6;
    options.OptimalityTolerance = 1e-6;
    options.MaxIterations       = 1000;

    cons = @(z) nonlcon(z, x, C);
    options.HessianFcn = @(z,lambda) hessian_fcn(z, x, lambda, C);

    wb = waitbar(0, 'Starting');
    for i = 1:size(P, 1)
        pt = P(i, :)';
        obj = @(z) objective(z, pt);
        z_opt = fmincon(obj, zeros(C.nz, 1), [],[],[],[],[],[], cons, options);
        V(i, :) = z_opt';
        waitbar(i/size(P, 1), wb, ...
            sprintf('Progress: %d %%', floor(i/size(P, 1)*100)));
    end
    close(wb);
end

function [fun, grad] = objective(z, pt)
    fun = sum((z - pt).^2);
    grad = 2 * (z - pt);
end

function [c, ceq, GC, GCeq] = nonlcon(z, x, C)
    [A, ~, dAdz, ~,~] = C.derivatives(x, z, zeros(C.nr, 1));

    c = A';
    ceq = [];

    GC = dAdz';
    GCeq = [];
end

function Hout = hessian_fcn(z, x, lambda, C)
    [~,~,~,~, d2Adzz_y] = C.derivatives(x, z, lambda.ineqnonlin);

    Hout = 2 * eye(C.nz) + d2Adzz_y;
    Hout = sparse(Hout);
end
