classdef IntegratorSE < DynamicalSystem
    % Integrator system on SE(n), n = 2 or 3.
    %
    % x: [n + n^2, 1] vector.
    % u: [3, 1] vector if n == 2 or [6, 1] vector if n == 3.
    % \dot{x} = u \in T_x SE(n).

    properties (Access = private)
        dim
    end

    methods
        function obj = IntegratorSE(n, x0, Au, bu)
            assert(n == 2 || n == 3);
            nx = n + n*n;
            if n == 2
                nu = 3;
            else
                nu = 6;
            end
            if isempty(x0)
                R0 = eye(n);
                x0 = [zeros(n, 1); R0(:)];
            end
            if isempty(Au) || isempty(bu)
                Au = [eye(nu); -eye(nu)];
                bu = [ones(nu, 1); ones(nu, 1)];
            end

            obj@DynamicalSystem(nx, nu, Au, bu);
            obj.dim = n;
            obj.x = x0;
        end

        function [f, g] = dyn(obj, x)
            f = zeros(obj.nx, 1);
            if obj.dim == 2
                ctheta = x(3);
                stheta = x(4);
                g = [eye(2), zeros(2, 1);
                    zeros(4, 2), [-stheta; ctheta; -ctheta; -stheta]];
            else
                R = reshape(x(4:end), 3, 3);
                kR = kron(eye(3), R);
                gw = [kR(:,6)-kR(:,8), kR(:,7)-kR(:,3), kR(:,2)-kR(:,4)];
                g = [eye(3), zeros(3,9); zeros(9,3), gw];
            end
        end

        function [x_new] = step(obj, dt, u)
            p = obj.x(1:obj.dim);
            R = reshape(obj.x(1+obj.dim:end), obj.dim, obj.dim);

            v = u(1:obj.dim);
            w = u(1+obj.dim:end);
            if obj.dim == 2
                w_hat = [0, -w; w, 0];
            else
                w_hat = [0, -w(3), w(2); w(3), 0, -w(1); -w(2), w(1), 0];
            end

            p_new = p + dt * v;
            R_new = R * expm(dt * w_hat);
            x_new = [p_new; R_new(:)];
        end
    end
end
