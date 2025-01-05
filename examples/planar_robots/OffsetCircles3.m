classdef OffsetCircles3 < AbstractConvexSet
    % 2d convex set which is intersection of 3 offset circles, with
    % parameters: (position, orientation).
    %
    % (p, theta): (position, orientation) \in R^3.
    % C = C1 \cap C2 \cap C3, where
    % Ci = {z: |z_inv - p_mat(i,:)|^2 <= R_mat(i)}, where z_inv = R' * (z - p).

    properties (SetAccess = immutable)
        p_mat
        R_arr
    end

    properties (Access = public)
        surf_pts = []
    end

    properties (Access = public)
        center = []
        radius = []
    end

    methods
        function obj = OffsetCircles3(p_mat, R_arr)
            % p_arr = (3, 2) mat.
            % R_arr = (3, 1) arr.
            obj = obj@AbstractConvexSet(3, 2, 3);

            obj.p_mat = p_mat;
            obj.R_arr = R_arr;
        end

        function cons = A(obj, x, z)
            cons = zeros(3, 1);
            for i = 1:3
                cons(i) = offset_circle(x, z, [obj.p_mat(i, :)'; obj.R_arr(i)]);
            end
        end

        function [A, dAdx, dAdz, d2Adxz_y, d2Adzz_y] = derivatives(obj, x, z, y)
            [A1, dAdx1, dAdz1, d2Adxz1, d2Adzz1] = offset_circle(x, z, ...
                [obj.p_mat(1, :)'; obj.R_arr(1)]);
            [A2, dAdx2, dAdz2, d2Adxz2, d2Adzz2] = offset_circle(x, z, ...
                [obj.p_mat(2, :)'; obj.R_arr(2)]);
            [A3, dAdx3, dAdz3, d2Adxz3, d2Adzz3] = offset_circle(x, z, ...
                [obj.p_mat(3, :)'; obj.R_arr(3)]);
            A = [A1; A2; A3];
            dAdx = [dAdx1; dAdx2; dAdx3];
            dAdz = [dAdz1; dAdz2; dAdz3];
            d2Adxz_y = y(1) * d2Adxz1 + y(2) * d2Adxz2 + y(3) * d2Adxz3;
            d2Adzz_y = y(1) * d2Adzz1 + y(2) * d2Adzz2 + y(3) * d2Adzz3;
        end

        function [obj] = plot_surf(obj, x, hdl, fc, fa, ~)
            p = x(1:2);
            R = [cos(x(3)), -sin(x(3)); sin(x(3)), cos(x(3))];

            if isempty(obj.surf_pts)
                obj.surf_pts = get_surf_points(obj, zeros(3, 1), 250);
            end
            V_ = R * obj.surf_pts' + p * ones(1, size(obj.surf_pts, 1));

            fill(hdl, V_(1, :), V_(2, :), fc, 'FaceAlpha', fa, ...
                'EdgeColor', [0, 0, 0]);
        end
    end
end
