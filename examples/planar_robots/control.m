function [U, dist_struct, solve_time] = control(robots, dist_struct, xf, clfs, ...
    dt, a_cbf, a_clf, w_clf, dist_margin)
    persistent U_idx m P_qp q_qp A_qp l_qp u_qp settings

    nrobots = numel(robots);
    ncbf = nrobots * (nrobots - 1) / 2;
    nclf = nrobots;

    %% Initialize persistent variables.
    if isempty(U_idx)
        U_idx = zeros(nrobots + 1, 1);
        for i = 1:nrobots
            U_idx(i+1) = U_idx(i) + robots{i}.system.nu;
        end
    end
    nvars = U_idx(end) + nclf;
    if isempty(m)
        m = osqp;
        settings = m.default_settings();
        settings.verbose = false;

        P_qp = 2 * blkdiag(eye(U_idx(end)), w_clf * eye(nclf));
        P_qp = sparse(P_qp);
        q_qp = zeros(nvars, 1);

        A_clf = zeros(nclf, nvars);
        l_clf = -Inf * ones(nclf, 1);
        u_clf = Inf * ones(nclf, 1);
        ncons_in = 0;
        for i = 1:nrobots
            A_clf(i, U_idx(i)+1:U_idx(i+1)) = ones(1, U_idx(i+1)-U_idx(i));
            A_clf(i, U_idx(end) + i) = -1;
            ncons_in = ncons_in + size(robots{i}.system.Au, 1);
        end
        A_cbf = zeros(ncbf, nvars);
        l_cbf = -Inf * ones(ncbf, 1);
        u_cbf = Inf * ones(ncbf, 1);
        A_in = zeros(ncons_in, nvars);
        l_in = -Inf * ones(ncons_in, 1);
        u_in = zeros(ncons_in, 1);
        i_in = 0;
        i_cbf = 1;
        for i = 1:nrobots
            si = size(robots{i}.system.Au, 1);
            A_in(i_in+1:i_in+si, U_idx(i)+1:U_idx(i+1)) = robots{i}.system.Au;
            u_in(i_in+1:i_in+si) = robots{i}.system.bu;
            i_in = i_in + si;
            for j = i+1:nrobots
                A_cbf(i_cbf, U_idx(i)+1:U_idx(i+1)) = ...
                    ones(1, U_idx(i+1)-U_idx(i));
                A_cbf(i_cbf, U_idx(j)+1:U_idx(j+1)) = ...
                    ones(1, U_idx(j+1)-U_idx(j));
                i_cbf = i_cbf + 1;
            end
        end
        A_qp = sparse([A_clf; A_cbf; A_in]);
        l_qp = [l_clf; l_cbf; l_in];
        u_qp = [u_clf; u_cbf; u_in];

        m.setup(P_qp, q_qp, A_qp, l_qp, u_qp, settings);
        m.solve();
    end

    %% Solve CBF-QP.

    f = cell(nrobots, 1);
    g = cell(nrobots, 1);
    % Update CLF terms.
    for i = 1:nrobots
        [V, DV] = clfs{i}(robots{i}.system.x, xf{i});
        [f{i}, g{i}] = robots{i}.system.dyn(robots{i}.system.x);
        A_qp(i, U_idx(i)+1:U_idx(i+1)) = DV * g{i};%#ok
        u_qp(i) = -a_clf * V - DV * f{i};
    end

    % Update CBF terms.
    idx = nclf + 1;
    for i = 1:nrobots
        for j = i+1:nrobots
            h = dist_struct{i, j}.dist2 - dist_margin;
            nr1 = dist_struct{i, j}.nr1;
            nr2 = dist_struct{i, j}.nr2;
            zopt = dist_struct{i, j}.zopt;
            yopt = dist_struct{i, j}.yopt;
            C1 = robots{i}.sets{1};
            C2 = robots{j}.sets{1};
            x1 = robots{i}.system.x;
            x2 = robots{j}.system.x;
            [~, dAdx1, ~,~,~] = C1.derivatives(x1, zopt(1:2), zeros(nr1, 1));
            [~, dAdx2, ~,~,~] = C2.derivatives(x2, zopt(3:4), zeros(nr2, 1));
            Lfh = yopt(1:nr1)' * dAdx1 * f{i} + yopt(1+nr1:end)' * dAdx2 * f{j};
            u_qp(idx) = a_cbf * h - Lfh;
            A_qp(idx, U_idx(i)+1:U_idx(i+1)) = -yopt(1:nr1)' * dAdx1 * g{i};%#ok
            A_qp(idx, U_idx(j)+1:U_idx(j+1)) = -yopt(1+nr1:end)' * dAdx2 * g{j};%#ok
            idx = idx + 1;
        end
    end

    % Update OSQP, and resolve.
    % m.update('u', u_qp, 'Ax', nonzeros(A_qp));
    m = osqp;
    m.setup(P_qp, q_qp, A_qp, l_qp, u_qp, settings);
    t_control_qp = tic;
    res = m.solve();
    qp_solve_time = toc(t_control_qp);
    if res.info.status_val ~= 1
        disp('error');
    end

    inputs = res.x(1:U_idx(end));
    U = cell(nrobots, 1);
    solve_time = cell(nrobots, 1);
    for i = 1:nrobots
        U{i} = inputs(U_idx(i)+1:U_idx(i+1));
        solve_time{i} = [0; qp_solve_time];
    end

    %% Update distances.
    for i = 1:nrobots
        for j = 1:nrobots
            if i == j
                continue
            end
            C1 = robots{i}.sets{1};
            C2 = robots{j}.sets{1};
            x1 = robots{i}.system.x;
            x2 = robots{j}.system.x;
            dx1 = f{i} + g{i} * U{i};
            dx2 = f{j} + g{j} * U{j};
            zopt = dist_struct{i, j}.zopt;
            yopt = dist_struct{i, j}.yopt;
            tic;
            [dist_struct{i, j}.dist2, dist_struct{i, j}.zopt, dist_struct{i, j}.yopt, ~] = ...
                minimum_distance_step(dt, x1, dx1, C1, x2, dx2, C2, zopt, yopt, 'euler');
            dist_solve_time_ = toc;
            solve_time{i}(1) = solve_time{i}(1) + dist_solve_time_;
        end
    end
end
