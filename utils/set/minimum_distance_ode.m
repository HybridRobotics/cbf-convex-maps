function [dz_opt, dy_opt, out] = ...
    minimum_distance_ode(x1, dx1, C1, x2, dx2, C2, z_opt, y_opt)
    % Computes the directional derivatives (dz_opt, dy_opt) of (z_opt,
    % y_opt) along dx.
    % Note: At least one of the convex sets must be strictly convex.
    %
    % Inputs:
    %   x1, x2: Current parameters for C1 and C2.
    %   dx1, dx2: Rate of change of parameters x1 and x2.
    %   C1, C2: Convex sets, at least one of them strictly convex.
    %   z_opt: Optimal primal solution for x=(x1, x2).
    %   y_opt: Optimal dual solution for x=(x1, x2).
    % Outputs:
    %   dz_opt: Directional derivative of z_opt at x along dx.
    %   dy_opt: Directional derivative of y_opt at x along dx.
    %   out: Struct for additional information.
    EPS = 1e-7; % Margin for index set calculations.
    ALPHA = 10; % Stability constant for the KKT conditions.

    nz = C1.nz; % = C2.nz;
    z1 = z_opt(1:nz);
    z2 = z_opt(1+nz:end);
    y1 = y_opt(1:C1.nr);
    y2 = y_opt(1+C1.nr:end);
    % Constraint derivatives.
    [A1, dAdx1, dAdz1, d2Adxz_y1, d2Adzz_y1] = C1.derivatives(x1, z1, y1);
    [A2, dAdx2, dAdz2, d2Adxz_y2, d2Adzz_y2] = C2.derivatives(x2, z2, y2);
    % Index sets.
    % J0c: Inactive primal constraints at z_opt.
    % J0 : Active primal constraints at z_opt.
    % J1 : Inactive (non-zero) dual variables.
    % J2e: Active primal constraints at z_opt, with zero dual.
    % J1c: Active (zero) dual variables.
    % Disjoint union of J0c, J1, and J2e is (1:C1.nr+C2.nr)'.
    J = (1:C1.nr+C2.nr)';
    J0c = J([A1; A2] < -EPS);
    % J0  = setdiff(J, J0c);
    % J1  = setdiff(find(y_opt > EPS), J0c);
    J1  = fast_setdiff(J(y_opt > EPS), J0c, C1.nr + C2.nr);
    % J2e = setdiff(J, union(J0c, J1));
    J2e = fast_2complement(J0c, J1, C1.nr + C2.nr);
    % J1c = union(J0c, J2e);

    A = [A1; A2];
    dAdx_dx = [dAdx1 * dx1; dAdx2 * dx2];
    dAdz = [dAdz1, zeros(C1.nr, nz);
        zeros(C2.nr, nz), dAdz2];
    % z-gradient of the Lagrangian.
    dLdz = 2 * [z1 - z2; z2 - z1] + dAdz' * y_opt;
    % x-derivative of z-gradeint of the Lagrangian.
    d2Ldxz_dx = [d2Adxz_y1 * dx1; d2Adxz_y2 * dx2];
    % z-hessian of the Lagrangian.
    d2Ldzz = [2 * eye(nz) + d2Adzz_y1, -2 * eye(nz);
        -2 * eye(nz), 2 * eye(nz) + d2Adzz_y2];

    % Case 1: J2e is empty.
    if isempty(J2e)
        out.J2e = 0;
        % Direct method.
        Q_mat = sparse([d2Ldzz, dAdz(J1, :)';
            -dAdz(J1, :), zeros(length(J1))]);
        V_vec = [-d2Ldxz_dx; dAdx_dx(J1)];
        kkt_vec = [dLdz; -A(J1)];
        sol = Q_mat \ (V_vec - ALPHA * kkt_vec);
        dz_opt = sol(1:2*nz);
        dy_opt = zeros(size(y_opt));
        dy_opt(J1)  = sol(1+2*nz:end);
        dy_opt(J0c) = -ALPHA * y_opt(J0c);

        % Factorization method (slow).
        % d2Ldzz_inv = decomposition(sparse(d2Ldzz), 'chol', 'upper');
        % dy_opt = zeros(size(y_opt));
        % dy_opt(J1) = mldivide(dAdz(J1, :) * (d2Ldzz_inv \ dAdz(J1, :)'), ...
        %     dAdx(J1, :) * dx + ALPHA * A(J1) - dAdz(J1, :) * (d2Ldzz_inv \ ...
        %     (d2Ldxz * dx + ALPHA * dLdz)));
        % dy_opt(J0c) = -ALPHA * y_opt(J0c);
        % dz_opt = -d2Ldzz_inv \ (dAdz(J1, :)' * dy_opt(J1) + d2Ldxz * dx + ...
        %     ALPHA * dLdz);

    % Case 2: J2e is non-empty.
    else
        out.J2e = 1;
        P_qp = sparse(d2Ldzz + 1e-2);
        q_qp = d2Ldxz_dx + ALPHA * (dLdz - dAdz(J0c, :)' * y_opt(J0c));
        A_qp = sparse([dAdz(J1, :); dAdz(J2e, :)]);
        l_qp = [-dAdx_dx(J1) - ALPHA * A(J1); -Inf * ones(length(J2e), 1)];
        u_qp = [l_qp(1:length(J1)); -dAdx_dx(J2e)];

        m = osqp;
        settings = m.default_settings();
        settings.time_limit = 1e-3; % [s].
        settings.verbose = false;
        try
            m.setup(P_qp, q_qp, A_qp, l_qp, u_qp, settings);
            results = m.solve();
        catch
            dz_opt = zeros(size(z_opt));
            dy_opt = zeros(size(y_opt));
            return;
        end

        if results.info.status_val == 1
            dz_opt = results.x;
            dy_opt = zeros(size(y_opt));
            dy_opt(J0c) = -ALPHA * y_opt(J0c);
            dy_opt(J1)  = results.y(1:length(J1));
            dy_opt(J2e) = results.y(1+length(J1):end);
        else
            dz_opt = zeros(size(z_opt));
            dy_opt = zeros(size(y_opt));
        end
    end
end
