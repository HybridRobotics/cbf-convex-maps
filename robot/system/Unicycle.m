classdef Unicycle < DynamicalSystem
    % Unicycle system in 2d.
    %
    % x: [3, 1] vector, = (p1, p2, theta).
    % u: [2, 1] vector, = (u1, u2).
    % \dot{x} = [u1 cos(theta); u1 sin(theta); u2].

    methods
        function obj = Unicycle(x0, Au, bu)
            if isempty(x0)
                x0 = zeros(3, 1);
            end
            if isempty(Au) || isempty(bu)
                Au = [eye(2); -eye(2)];
                bu = [1; 2*pi; 1; 2*pi];
            end

            obj@DynamicalSystem(3, 2, Au, bu);
            obj.x = x0;
        end

        function [f, g] = dyn(obj, x)
            f = zeros(obj.nx, 1);
            g = [cos(x(3)), 0; sin(x(3)), 0; 0, 1];
        end
    end
end
