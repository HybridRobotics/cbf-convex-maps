classdef DoubleIntegratorSE < DynamicalSystem
    % Double integrator system on SE(3).
    % 
    % x: [2*3 + 9, 1] vector: x = (p,v,R).
    % u: [3+3, 1] vector: u = (a,w).
    % \dot{x} = (v, a, R\hat{w}).

    methods
        function obj = DoubleIntegratorSE(x0, Au, bu)
            nx = 2*3 + 9;
            nu = 6;
            if isempty(x0)
                R0 = eye(3);
                x0 = [zeros(6, 1); R0(:)];
            end
            if isempty(Au) || isempty(bu)
                Au = [eye(nu); -eye(nu)];
                bu = [ones(nu, 1); ones(nu, 1)];
            end
            
            obj@DynamicalSystem(nx, nu, Au, bu);
            obj.x = x0;
        end
        
        function [f, g] = dyn(~, x)
            R = reshape(x(7:end), 3, 3);

            f = [x(4:6); zeros(12, 1)];
            gw = [zeros(3, 1), -R(:,3), R(:,2);
                R(:,3), zeros(3, 1), -R(:,1);
                -R(:,2), R(:,1), zeros(3, 1)];
            g = [zeros(3, 6); eye(3), zeros(3); zeros(9, 3), gw];
        end

        function [x_new] = step(obj, dt, u)
            p = obj.x(1:3);
            v = obj.x(4:6);
            R = reshape(obj.x(7:end), 3, 3);

            a = u(1:3);
            w = u(4:6);
            w_hat = [0, -w(3), w(2); w(3), 0, -w(1); -w(2), w(1), 0];

            v_new = v + dt * a;
            p_new = p + dt/2 * (v + v_new);
            R_new = R * expm(dt * w_hat);
            x_new = [p_new; v_new; R_new(:)];
        end
    end
end
