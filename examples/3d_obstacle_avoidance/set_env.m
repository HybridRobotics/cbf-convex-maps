function [robots, xf, tracking_clf] = set_env()
    NROBOTS = 8; % per team.
    sidelength = ceil((NROBOTS)^(1/3));
    R = 10; % minimum inter-robot distance.
    SEP = 10; % separation between the two teams.
    TMAX = 3;
    ALPHA_P = 0.5; % 0.5, 1

    rng(1,"twister");

    % Assign dynamics.
    systems = cell(2 * NROBOTS, 1);
    xf = cell(2 * NROBOTS, 1);
    for i = 1:2*NROBOTS
        [i1, i2, i3] = ind2sub(sidelength * ones(1, 3), mod(i-1, NROBOTS) + 1);
        p0 = [R * i1; R * i2; R * i3];
        sep_p = [SEP; 0; 0];
        R0 = eye(3);

        if i <= NROBOTS
            x0 = [p0 - sep_p; zeros(3, 1); R0(:)];
            xf{i} = [p0 + sep_p; zeros(3, 1); R0(:)];
        else
            x0 = [p0 + sep_p; zeros(3, 1); R0(:)];
            xf{i} = [p0 - sep_p; zeros(3, 1); R0(:)];
        end
        Au = [eye(6); -eye(6)];
        bu_ = [ones(3, 1); pi/2 * ones(3, 1)];
        bu = [bu_; bu_];
        systems{i} = DoubleIntegratorSE(x0, Au, bu);
        % systems{i}.check_system();
    end
    disp('Assigned dynamics');

    % Assign convex sets.
    sets = cell(2 * NROBOTS, 1);
    for i = 1:2*NROBOTS
        nrandpts = randi([10, 15]); % [7, 9].
        j = 0;
        V_ = zeros(nrandpts, 3);
        while true
            s2_pt = 2 * rand(1, 3) - ones(1, 3);
            if norm(s2_pt, 2) < 1e-3
                continue;
            end
            j = j + 1;
            s2_pt = s2_pt / norm(s2_pt, 2);
            rad = (rand() + 1) / 2;
            V_(j,:) = rad * s2_pt;
            if j == nrandpts
                break
            end
        end
        P_ = Polyhedron(V_);
        P_.minHRep();
        P.A_mat = P_.A;
        P.b_vec = P_.b;
        sets{i} = SmoothPolytope4D(P, ALPHA_P, TMAX);
        sets{i}.radius = R + 2;
        sets{i}.center = [0; 0; 0];
    end
    disp('Assigned convex sets');

    % Plot initial state.
    figure();
    ax = gca;
    for i = 1:2*NROBOTS
        hold on
        sets{i} = sets{i}.plot_surf(systems{i}.x, ax, 'r', 1, []);
    end
    axis equal

    % Set robots.
    robots = cell(2 * NROBOTS, 1);
    for i = 1:2*NROBOTS
        robots{i} = Robot(systems{i}, {sets{i}});
    end

    tracking_clf = get_clf('double-integrator');
end


%%
function [func] = get_clf(type)
    switch type
        case 'double-integrator'
            func = @(x, xd) doubleintegrator_clf(x, xd);
        case 'simple-quadrotor'
            return
    end
end

function [V, DV] = doubleintegrator_clf(x, xd)
    k_R = 4;
    k_vv = 2;
    k_pv = 1;

    p = x(1:3);
    v = x(4:6);
    R = reshape(x(7:end), 3, 3);
    pd = xd(1:3);
    vd = xd(4:6);
    Rd = reshape(xd(7:end), 3, 3);

    V = 1/2 * sum((p - pd).^2) + 1/2 * k_vv * sum((v - vd).^2) + ...
        k_pv * (p - pd)' * (v - vd) + k_R/2 * trace(eye(3) - Rd' * R);
    DV = [(p - pd)' + k_pv * (v - vd)', k_pv * (p - pd)' + k_vv * (v - vd)', ...
        -k_R/2 * Rd(:)'];
end

function [V, DV] = simplequadrotor_clf(x, xd)
    %
end
