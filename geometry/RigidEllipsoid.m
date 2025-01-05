classdef RigidEllipsoid < AbstractConvexSet
    % nz-dimensional ellipsoid with parameters: (position, orientation).
    %
    % (p, R): (position, orientation).
    % Q: Positive-definite weight matrix.
    % C = {z: (z-p)^T R Q R^T (z-p) <= 1}.

    properties
        Q
    end

    properties (Access = public)
        surf_pts = []
    end

    methods
        function obj = RigidEllipsoid(nz, Q)
            nx = nz + nz * nz;
            obj = obj@AbstractConvexSet(nx, nz, 1);

            assert(isequal(size(Q), [nz, nz]));
            try chol(Q);
            catch
                disp('Q matrix is not symmetric positive definite');
            end
            obj.Q = Q;
        end

        function cons = A(obj, x, z)
            p = x(1:obj.nz);
            % R is vectorized column-first.
            R = reshape(x(obj.nz+1:end), obj.nz, obj.nz);

            cons = (z - p)' * R * obj.Q * R' * (z-p) - 1;
        end

        function [A, dAdx, dAdz, d2Adxz_y, d2Adzz_y] = derivatives(obj, x, z, y)
            p = x(1:obj.nz);
            % R is vectorized column-first.
            R = reshape(x(obj.nz+1:end), obj.nz, obj.nz);

            % Note: A x = \sum_i A^i x_i = \sum_i (x_i I) A^i,
            %           = kron(x', I) vec(A).
            %       A'y = [y' A^1; y' A^2; ...] = kron(I, y') vec(A).
            A = (z - p)' * R * obj.Q * R' * (z-p) - 1;
            dAdz = 2 * (z - p)' * R * obj.Q * R';
            dAdx = [-dAdz, ...
                2 * (z - p)' * R * obj.Q * kron(eye(obj.nz), (z - p)')];
            d2Adzz_y = y(1) * 2 * R * obj.Q * R';
            d2Adxz_y = [-d2Adzz_y, ...
                y(1) * kron(2 * (z - p)' * R * obj.Q, eye(obj.nz)) + ...
                y(1) * 2 * R * obj.Q * kron(eye(obj.nz), (z - p)')];
        end

        function [obj] = plot_surf(obj, x, hdl, fc, fa, ea)
            p = x(1:obj.nz);
            % R is vectorized column-first.
            R = reshape(x(obj.nz+1:end), obj.nz, obj.nz);

            if obj.nz == 2
                nsurf = 25;

                if isempty(obj.surf_pts)
                    [v, lambda] = eig(obj.Q);
                    pts_ = (0:nsurf-1) / nsurf * 2*pi;
                    obj.surf_pts = v * ...
                        [1/sqrt(lambda(1,1)) * cos(pts_);
                        1/sqrt(lambda(2, 2)) * sin(pts_)];
                end
                surf_ = R * obj.surf_pts + p * ones(1, size(obj.surf_pts, 2));
                fill(hdl, surf_(1, :), surf_(2, :), fc, 'FaceAlpha', fa, ...
                    'EdgeColor', [0, 0, 0]);
            elseif obj.nz == 3
                nsurf = 25;
                step = 5;

                if isempty(obj.surf_pts)
                    [v, lambda] = eig(obj.Q);
                    [Xm_, Ym_, Zm_] = ellipsoid(0, 0, 0, 1/sqrt(lambda(1,1)), ...
                        1/sqrt(lambda(2,2)), 1/sqrt(lambda(3,3)), nsurf);
                    obj.surf_pts = v * [Xm_(:)'; Ym_(:)'; Zm_(:)'];
                end
                surf_ = R * obj.surf_pts + p * ones(1, size(obj.surf_pts, 2));
                Xm_ = reshape(surf_(1, :), [nsurf+1, nsurf+1]);
                Ym_ = reshape(surf_(2, :), [nsurf+1, nsurf+1]);
                Zm_ = reshape(surf_(3, :), [nsurf+1, nsurf+1]);
                hold on
                surf(hdl, Xm_, Ym_, Zm_, 'FaceColor', fc, 'FaceAlpha', fa, ...
                    'EdgeColor', 'none');
                plot3(hdl, Xm_(:, 1:step:end), Ym_(:, 1:step:end), ...
                    Zm_(:, 1:step:end), 'Color', [0, 0, 0, ea]);
                plot3(hdl, Xm_(1:step:end, :)', Ym_(1:step:end, :)', ...
                    Zm_(1:step:end, :)', 'Color', [0, 0, 0, ea]);
                hold off
            else
                error('Plotting for nz not 2 or 3 is not supported');
            end
        end
    end
end
