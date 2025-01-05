classdef UnicycleSE2 < DynamicalSystem
    % Unicycle system in 2d with SE(2) representation.
    %
    % x: [6, 1] vector, = (p, R).
    % u: [2, 1] vector, = (u1, u2).
    % \dot{x} = [R * [1; 0] * u1; vec(R * hat(u2))].

    methods
        function obj = UnicycleSE2(x0, Au, bu)
            if isempty(x0)
                R0 = eye(2);
                x0 = [zeros(2, 1); R0(:)];
            end
            if isempty(Au) || isempty(bu)
                Au = [eye(2); -eye(2)];
                bu = [1; 2*pi; 1; 2*pi];
            end

            obj@DynamicalSystem(6, 2, Au, bu);
            obj.x = x0;
        end

        function [f, g] = dyn(obj, x)
            ctheta = x(3);
            stheta = x(4);

            f = zeros(obj.nx, 1);
            g = [ctheta, 0; stheta, 0;
                zeros(4, 1), [-stheta; ctheta; -ctheta; -stheta]];
        end

        function [x_new] = step(obj, dt, u)
            p = obj.x(1:2);
            R = reshape(obj.x(3:end), 2, 2);

            v = u(1);
            w = u(2);
            w_hat = [0, -w; w, 0];

            p_new = p + dt * v * R(:, 1);
            R_new = R * expm(dt * w_hat);
            x_new = [p_new; R_new(:)];
        end
    end
end
