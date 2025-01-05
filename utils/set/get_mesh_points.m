function [V] = get_mesh_points(C, x, nstep, npts)
    % Find mesh points on the surface of a 3d convex set.
    %
    % Inputs:
    %   C: Convex set, C.radius and C.center should be defined.
    %   x: State of C.
    %   nstep: Number of contours along each axis.
    %   npts: Number of points on each contour.
    % Outputs:
    %   V: [npts, nz, nstep, nz] matrix of mesh points. 2nd-axis is the dim.

    if nargin < 3
        nstep = 3;
        npts = 26;
    end

    assert(C.nz == 3, 'Only nz = 3 is supported');

    options = optimoptions('fmincon', 'Display', 'off', ...
    'Algorithm', 'interior-point');
    options.SubproblemAlgorithm = 'factorization';
    options.SpecifyConstraintGradient = true;
    options.SpecifyObjectiveGradient  = true;
    options.ConstraintTolerance = 1e-6;
    options.OptimalityTolerance = 1e-6;
    options.MaxIterations       = 1e4;

    cons = @(z) nonlcon(z, x, C);

    % Find end points.
    eye_mat = eye(3);
    limit_l = zeros(3, 1);
    limit_u = zeros(3, 1);

    temp_ = C.radius * ones(3, 1);
    options.HessianFcn = @(z,lambda) hessian_fcn('end-pts', z, x, lambda, C);
    for i = 1:3
        obj = @(z) objective('end-pts', eye_mat(:, i), z, []);
        [~, limit_l(i), flag] = fmincon(obj, zeros(3, 1), [],[],[],[], ...
            -temp_, temp_, cons, options);
        if flag ~= 1
            error('');
        end
        obj = @(z) objective('end-pts', -eye_mat(:, i), z, []);
        [~, limit_, flag] = fmincon(obj, zeros(3, 1), [],[],[],[], ...
            -temp_, temp_, cons, options);
        if flag ~= 1
            error('');
        end
        limit_u(i) = -limit_;
    end
    steps = (0:nstep+1) / (nstep + 1);
    steps = steps(2:end-1);
    coords = limit_l * ones(size(steps)) + (limit_u - limit_l) * steps;

    % Find mesh points.
    V = zeros(npts, 3, nstep, 3);
    angles = (0:npts-1)' / (npts-1) * 2*pi;
    circ = cell(3, 1);
    circ{1} = C.radius * [0 * angles, cos(angles), sin(angles)];
    circ{2} = C.radius * [sin(angles), 0 * angles, cos(angles)];
    circ{3} = C.radius * [cos(angles), sin(angles), 0 * angles];

    wb = waitbar(0, 'Starting');
    wb_iter = 3 * nstep * npts;
    for i1 = 1:3 % Axis.
        axis = eye_mat(:, i1);
        for i2 = 1:nstep % Contour step.
            center = C.center .* (ones(3, 1) - axis) + coords(i1, i2);
            coord = coords(i1, i2);
            pts = circ{i1} + ones(npts, 1) * center';
            for i3 = 1:npts % Point.
                iter = (npts*nstep) * (i1-1) + (npts) * (i2-1) + i3;
                pt = pts(i3, :)';
                obj = @(z) objective('proj', [], z, pt);
                [z_opt, ~, flag] = fmincon(obj, zeros(3, 1), [],[], axis', ...
                    coord, [], [], cons, options);
                if flag ~= 1
                    continue;
                end
                V(i3, :, i2, i1) = z_opt';
                waitbar(iter/wb_iter, wb, ...
                    sprintf('Progress: %d %%', floor(iter/wb_iter*100)));
            end % i3
        end % i2
    end % i1
    close(wb);
end

function [fun, grad] = objective(type, axis, z, pt)
    if strcmp(type, 'end-pts')
        fun = axis' * z;
        grad = axis;
    elseif strcmp(type, 'proj')
        fun = sum((z - pt).^2);
        grad = 2 * (z - pt);
    end
end

function [c, ceq, GC, GCeq] = nonlcon(z, x, C)
    [A, ~, dAdz, ~,~] = C.derivatives(x, z, zeros(C.nr, 1));

    c = A';
    GC = dAdz';

    ceq = [];
    GCeq = [];
end

function Hout = hessian_fcn(type, z, x, lambda, C)
    [~,~,~,~, d2Adzz_y] = C.derivatives(x, z, lambda.ineqnonlin);

    if strcmp(type, 'end-pts')
        Hout = d2Adzz_y;
    elseif strcmp(type, 'proj')
        Hout = 2 * eye(3) + d2Adzz_y;
    end
    Hout = sparse(Hout);
end
