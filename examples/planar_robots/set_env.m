function [robots, xf, clfs] = set_env()
    nrobots = 5;
    R = 9;

    rng(1,"twister");

    % Assign dynamics.
    systems = cell(nrobots, 1);
    xf = cell(nrobots, 1);
    clfs = cell(nrobots, 1);
    for i = 1:nrobots
        p0 = [R * cos(2*pi/nrobots*(i-1)); R * sin(2*pi/nrobots*(i-1))];
        R0 = eye(2);
        x0 = [p0; R0(:)];
        xf{i} = [-p0; R0(:)];
        if 2*i <= nrobots
            Au = [eye(3); -eye(3)];
            bu = [0.5; 0.5; pi/2; 0.5; 0.5; pi/2] + ...
                [0.25; 0.25; pi/4; 0.25; 0.25; pi/4] .* (1 - 2*rand(6, 1));
            systems{i} = IntegratorSE(2, x0, Au, bu);
            clfs{i} = @(x, xd) integrator_clf(x, xd);
            % systems{i}.check_system();
        else
            Au = [eye(2); -eye(2)];
            bu = [0.5; pi/2; 0.5; pi/2] + ...
                [0.25; pi/4; 0.25; pi/4] .* (1 - 2*rand(4, 1));
            systems{i} = UnicycleSE2(x0, Au, bu);
            clfs{i} = @(x, xd) unicycle_clf(x, xd);
            % systems{i}.check_system();
        end
    end
    disp('Assigned dynamics');

    % Assign convex sets.
    sets = cell(nrobots, 1);
    % sets{1} = Norm4Level(0.5);
    % sets{1}.radius = 2; sets{1}.center = zeros(2, 1);
    % sets{2} = OffsetCircles3([0.1, 0.1; -0.5, 0; 0, -0.2], 0.75 * [1; 1; 1]);
    % sets{2}.radius = 3; sets{2}.center = zeros(2, 1);
    nellipsoids = 1;
    for i = 1:nellipsoids
        sets{i} = RigidEllipsoid(2, 0.75 * [3, 1; 1, 2]);
    end
    for i = nellipsoids+1:nrobots
        nrandpts = 10;
        r_ = rand(nrandpts, 1);
        theta_ = 2*pi * rand(nrandpts, 1);
        V_ = r_ .* [cos(theta_), sin(theta_)];
        P_ = Polyhedron(V_);
        P_.minHRep();
        P.A_mat = P_.A;
        P.b_vec = P_.b;
        sets{i} = SmoothPolytope(2, P, 2.5);
        sets{i}.radius = 2;
        sets{i}.center = [0; 0];
    end
    disp('Assigned convex sets');

    % Plot initial state.
    figure();
    ax = gca;
    for i = 1:nrobots
        hold on
        sets{i} = sets{i}.plot_surf(systems{i}.x, ax, 'r', 1, []);
    end

    % Set robots.
    robots = cell(nrobots, 1);
    for i = 1:nrobots
        robots{i} = Robot(systems{i}, sets(i));
    end
end

function [V, DV] = integrator_clf(x, xd)
    k_clf = 10;

    p = x(1:2);
    R = reshape(x(3:end), 2, 2);
    pd = xd(1:2);
    Rd = reshape(xd(3:end), 2, 2);

    V = 1/2 * sum((p - pd).^2) + k_clf/2 * trace(eye(2) - Rd' * R);
    DV = [(p - pd)', -k_clf/2 * Rd(:)'];
end

function [V, DV] = unicycle_clf(x, xd)
    k_clf = 10;

    p = x(1:2);
    R = reshape(x(3:end), 2, 2);
    pd = xd(1:2);
    vd = pd - p;
    if norm(vd) < 1e-6
        Rd = reshape(xd(3:end), 2, 2);
    else
        nd = vd / norm(vd);
        td = [-nd(2); nd(1)];
        Rd = [nd, td];
    end

    V = 1/2 * sum((p - pd).^2) + k_clf/2 * trace(eye(2) - Rd' * R);
    DV = [(p - pd)', -k_clf/2 * Rd(:)'];
end
