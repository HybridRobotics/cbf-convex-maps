classdef Norm4Level < AbstractConvexSet
    % 2d convex set with parameters: (position, orientation).
    %
    % (p, theta): (position, orientation) \in R^3.
    % C = {z: z1_inv^4 + z2_inv^4 <= lev}, where z_inv = R' * (z - p).

    properties (SetAccess = immutable)
        lev
    end

    properties (Access = public)
        surf_pts = []
    end

    properties (Access = public)
        center = []
        radius = []
    end

    methods
        function obj = Norm4Level(lev)
            obj = obj@AbstractConvexSet(3, 2, 1);

            obj.lev = lev;
        end

        function cons = A(obj, x, z)
            cons = norm4_lev(x, z, obj.lev);
        end

        function [A, dAdx, dAdz, d2Adxz_y, d2Adzz_y] = derivatives(obj, x, z, y)
            [A, dAdx, dAdz, d2Adxz, d2Adzz] = norm4_lev(x, z, obj.lev);
            d2Adxz_y = y(1) * d2Adxz;
            d2Adzz_y = y(1) * d2Adzz;
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
