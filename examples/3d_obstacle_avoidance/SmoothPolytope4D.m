classdef SmoothPolytope4D < AbstractConvexSet
    % Smooth approximation of a 4D extension of a polytope set
    % using log-sum-exp, with parameters as (position, velocity, orientation).
    %
    % (p, v, R): (position, velocity, orientation).
    % P = struct('A_mat', A_mat, 'b_vec', b_vec).
    % alpha: tightness parameter.

    properties (SetAccess = immutable)
        A_mat
        b_vec
        alpha
        Tmax
    end

    properties (Access = public)
        % [-1, nz] matrix, Vertices of a surface mesh for the set when
        % p = 0, R = I.
        V = []

        center = [] % Analytic center of the set when p = 0, R = I.
        radius = [] % Upper bound on the radius of the set.
    end

    properties (Access = public)
        surf_pts = []
        mesh_pts = []
    end

    methods
        function obj = SmoothPolytope4D(P, alpha, Tmax)
            assert(SetUtils.is_polytope_solid(P.A_mat, P.b_vec), ...
                'Polyhedron is not solid');
            assert(size(P.A_mat, 2) == 3);

            obj = obj@AbstractConvexSet(2*3 + 9, 4, 1);
            obj.A_mat = P.A_mat;
            obj.b_vec = P.b_vec;
            obj.Tmax = Tmax;
            obj.alpha = alpha;
        end

        function cons = A(obj, x, z_)
            p = x(1:3);
            v = x(4:6);
            % R is vectorized column-first.
            R = reshape(x(7:end), 3, 3);

            Ax = [obj.A_mat * R', -obj.A_mat * R' * v * obj.Tmax;
                zeros(1, 3), 1;
                zeros(1, 3), -1];
            bx = [obj.b_vec + obj.A_mat * R' * p; 1; 0];
            bx = bx + ones(size(bx)) * log(size(bx, 1)) / obj.alpha;
            cons = 1/obj.alpha * logsumexp_a(obj.alpha * (Ax * z_ - bx));
        end

        function [A, dAdx, dAdz, d2Adxz_y, d2Adzz_y] = ...
                derivatives(obj, x, z_, y)
            p = x(1:3);
            v = x(4:6);
            R = reshape(x(7:end), 3, 3);
            z = z_(1:3);
            t = z_(4);

            Ax = [obj.A_mat * R', -obj.A_mat * R' * v * obj.Tmax;
                zeros(1, 3), 1;
                zeros(1, 3), -1];
            bx = [obj.b_vec + obj.A_mat * R' * p; 1; 0];
            bx = bx + ones(size(bx)) * log(size(bx, 1)) / obj.alpha;
            [lse, sm] = logsumexp_a(obj.alpha * (Ax * z_ - bx));

            % Note: A x = \sum_i A^i x_i = \sum_i (x_i I) A^i,
            %           = kron(x', I) vec(A).
            %       A'y = [y' A^1; y' A^2; ...] = kron(I, y') vec(A).
            A = 1/obj.alpha * lse;
            dAdz = sm' * Ax;
            temp_ = obj.A_mat * R';
            dAdx_ = [-temp_, -temp_ * t * obj.Tmax, ...
                obj.A_mat * kron(eye(3), (z - p - v * t * obj.Tmax)');
                zeros(2, 15)];
            dAdx = sm' * dAdx_;
            d2Adzz_y = y(1) * obj.alpha * Ax' * (diag(sm) - sm * sm') * Ax;
            temp_ = sm(1:end-2)' * obj.A_mat;
            d2Adxz = obj.alpha * Ax' * (diag(sm) - sm * sm') * dAdx_ + ...
                [zeros(3, 6), kron(temp_, eye(3));
                zeros(1, 3), -obj.Tmax * temp_ * R', ...
                -obj.Tmax * v' * kron(temp_, eye(3))];
            d2Adxz_y = y(1) * d2Adxz;
        end

        function [obj] = plot_surf(obj, x, hdl, fc, fa, ea)
            if isempty(obj.center) || isempty(obj.radius)
                error(strcat('Specify the center and radius of a sphere ', ...
                    'that encapsulates the object when at state (p,R)=(0,I)'));
            end

            p = x(1:3);
            R = reshape(x(7:end), 3, 3);
            if isempty(obj.surf_pts)
                R_ = eye(3);
                x_ = [zeros(3, 1); R_(:)];
                P_.A_mat = obj.A_mat; P_.b_vec = obj.b_vec;
                obj_ = SmoothPolytope(3, P_, obj.alpha);
                obj_.center = obj.center;
                obj_.radius = obj.radius;
                obj.surf_pts = get_surf_points(obj_, x_);
                obj.mesh_pts = get_mesh_points(obj_, x_);
            end
            V_ = R * obj.surf_pts' + p * ones(1, size(obj.surf_pts, 1));

            K = convhull(V_(1, :), V_(2, :), V_(3, :), 'Simplify', true);
            hold on
            trisurf(K, V_(1, :), V_(2, :), V_(3, :), 'FaceColor', fc, ...
                'FaceAlpha', fa, 'EdgeColor', 'none', 'Parent', hdl, ...
                'LineWidth', 0.2);
            % trisurf(K, V_(1, :), V_(2, :), V_(3, :), 'FaceColor', fc, ...
            %     'FaceAlpha', fa, 'EdgeColor', 'k', 'Parent', hdl, ...
            %     'LineWidth', 0.2);
            mesh = obj.mesh_pts;
            for i = 1:size(mesh, 3)
                for j = 1:size(mesh, 4)
                    pts = mesh(:, :, i, j) * R' + ...
                        ones(size(mesh, 1), 1) * p';
                    plot3(hdl, pts(:, 1), pts(:, 2), pts(:, 3), ...
                        'Color', [0, 0, 0, ea]);
                end
            end
            hold off
        end
    end
end
