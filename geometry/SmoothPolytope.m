classdef SmoothPolytope < AbstractConvexSet
    % Smooth approximation of a polytope set using log-sum-exp, with
    % parameters as (position, orientation).
    %
    % (p, R): (position, orientation).
    % P = struct('A_mat', A_mat, 'b_vec', b_vec).
    % alpha: tightness parameter.
    % C = {z: 1/alpha log (\sum_i exp(alpha(A_mat_i R' (z - p) - b_vec_i)))}.

    properties (SetAccess = immutable)
        A_mat
        b_vec
        alpha
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
        function obj = SmoothPolytope(nz, P, alpha)
            assert(SetUtils.is_polytope_solid(P.A_mat, P.b_vec), ...
                'Polyhedron is not solid');
            assert(size(P.A_mat, 2) == nz);

            obj = obj@AbstractConvexSet(nz + nz * nz, nz, 1);
            obj.A_mat = P.A_mat;
            obj.b_vec = P.b_vec + ...
                ones(size(P.b_vec)) * log(size(P.b_vec, 1)) / alpha;
            obj.alpha = alpha;
        end

        function cons = A(obj, x, z)
            p = x(1:obj.nz);
            % R is vectorized column-first.
            R = reshape(x(obj.nz+1:end), obj.nz, obj.nz);

            Ax = obj.A_mat * R';
            bx = obj.b_vec + Ax * p;
            cons = 1/obj.alpha * logsumexp_a(obj.alpha * (Ax * z - bx));
        end

        function [A, dAdx, dAdz, d2Adxz_y, d2Adzz_y] = derivatives(obj, x, z, y)
            p = x(1:obj.nz);
            R = reshape(x(obj.nz+1:end), obj.nz, obj.nz);

            Ax = obj.A_mat * R';
            bx = obj.b_vec + Ax * p;
            [lse, sm] = logsumexp_a(obj.alpha * (Ax * z - bx));

            % Note: A x = \sum_i A^i x_i = \sum_i (x_i I) A^i,
            %           = kron(x', I) vec(A).
            %       A'y = [y' A^1; y' A^2; ...] = kron(I, y') vec(A).
            A = 1/obj.alpha * lse;
            dAdz = sm' * Ax;
            dAdx = [-dAdz, sm' * obj.A_mat * kron(eye(obj.nz), (z - p)')];
            d2Adzz_y = y(1) * obj.alpha * Ax' * (diag(sm) - sm * sm') * Ax;
            d2Adxz_y = [-d2Adzz_y, ...
                y(1) * obj.alpha * Ax' * (diag(sm) - sm * sm') * obj.A_mat * ...
                kron(eye(obj.nz), (z - p)') + ...
                y(1) * kron(sm' * obj.A_mat, eye(obj.nz))];
        end

        function [obj] = plot_surf(obj, x, hdl, fc, fa, ea)
            if obj.nz ~= 2 && obj.nz ~= 3
                error('Plotting for nz not 2 or 3 is not supported');
            end
            if isempty(obj.center) || isempty(obj.radius)
                error(strcat('Specify the center and radius of a sphere ', ...
                    'that encapsulates the object when at state (p,R)=(0,I)'));
            end

            p = x(1:obj.nz);
            R = reshape(x(obj.nz+1:end), obj.nz, obj.nz);
            if isempty(obj.surf_pts)
                R_ = eye(obj.nz);
                x_ = [zeros(obj.nz, 1); R_(:)];
                obj.surf_pts = get_surf_points(obj, x_);
                if obj.nz == 3
                    obj.mesh_pts = get_mesh_points(obj, x_);
                end
            end
            V_ = R * obj.surf_pts' + p * ones(1, size(obj.surf_pts, 1));

            if obj.nz == 2
                fill(hdl, V_(1, :), V_(2, :), fc, 'FaceAlpha', fa, ...
                    'EdgeColor', [0, 0, 0]);
            elseif obj.nz == 3
                K = convhull(V_(1, :), V_(2, :), V_(3, :), 'Simplify', true);
                hold on
                trisurf(K, V_(1, :), V_(2, :), V_(3, :), 'FaceColor', fc, ...
                    'FaceAlpha', fa, 'EdgeColor', 'none', 'Parent', hdl);
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
end
