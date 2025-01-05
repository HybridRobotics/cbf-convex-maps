classdef (Abstract) DynamicalSystem < handle
    % Abstract dynamical system.
    %
    % x: State variable, x \in R^nx.
    % u: Input variable, u \in U \subset R^nu.
    % Continuous-time dynamics: \dot{x} = f(x) + g(x) u.
    % Input constraints: U = {Au u <= bu}.

    properties (SetAccess = immutable)
        nx
        nu

        Au
        bu
    end

    properties (Access = public)
        x
    end

    methods (Access = public)
        function obj = DynamicalSystem(nx, nu, Au, bu)
            obj.nx = nx;
            obj.nu = nu;
            obj.Au = Au;
            obj.bu = bu;
        end
    end

    methods (Abstract)
        % Continuous-time dynamics functions.
        % \dot{x} = f(x) + g(x) u.
        % Inputs:
        %   x: [nx, 1] vector, State variable
        % Outputs:
        %   f: [nx, 1] vector, Flow vector at x, f(x).
        %   g: [nx, nu] matrix, Input matrix at x, g(x).
        dyn(obj, x)
    end

    methods
        % Setter function for state variable x.
        function [] = set_state(obj, x)
            obj.x = x;
        end

        % Computes and updates x after input u is applied for dt time.
        % Inputs:
        %   dt: scalar, Time interval of update.
        %   u: [nu, 1] vector, Input.
        % Outputs:
        %   x_new: [nx, 1] vector, state after dynamics update.
        function [x_new] = step(obj, dt, u)
            f, g = obj.dyn(obj.x);
            dx = f + g * u;
            x_new = obj.x + dx * dt;
            obj.x = x_new;
        end

        function [] = check_system(obj)
            disp(' ');
            disp(['Class: ' class(obj) ' < ' ...
                strjoin(superclasses(class(obj)), ', ') ':']);
            disp(['State dim: [' num2str(obj.nx) ' 1]'])
            disp(['Input dim: [' num2str(obj.nu) ' 1]'])
            if ~isequal(size(obj.x), [obj.nx, 1])
                disp(['size(obj.x) is not [' num2str(obj.nx) ' 1]; ' ...
                    'got [' num2str(size(obj.x)) ']']);
            end
            [f, g] = obj.dyn(obj.x);
            if ~isequal(size(f), [obj.nx, 1])
                disp(['size(f) is not [' num2str(obj.nx) ' 1]; ' ...
                    'got [' num2str(size(f)) ']']);
            end
            if ~isequal(size(g), [obj.nx, obj.nu])
                disp(['size(g) is not [' num2str([obj.nx, obj.nu]) ']; ' ...
                    'got [' num2str(size(g)) ']']);
            end
            if size(obj.Au, 2) ~= obj.nu || size(obj.Au, 1) ~= size(obj.bu, 1)
                disp('Input constraint size is incorrect');
            end
        end
    end
end
