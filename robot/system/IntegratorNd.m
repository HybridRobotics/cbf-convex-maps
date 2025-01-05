classdef IntegratorNd < DynamicalSystem
    % Integrator system in nx-d.
    %
    % x: [nx, 1] vector.
    % u: [nx, 1] vector.
    % \dot{x} = u.

    methods
        function obj = IntegratorNd(nx, x0, Au, bu)
            if isempty(nx)
                nx = 2;
            end
            if isempty(x0)
                x0 = zeros(nx, 1);
            end
            if isempty(Au) || isempty(bu)
                Au = [eye(nx); -eye(nx)];
                bu = [ones(nx, 1); ones(nx, 1)];
            end

            obj@DynamicalSystem(nx, nx, Au, bu);
            obj.x = x0;
        end

        function [f, g] = dyn(obj, ~)
            f = zeros(obj.nx, 1);
            g = eye(obj.nu);
        end
    end
end
