classdef (Abstract) AbstractConvexSet < handle
    % Parametric convex set representation.
    %
    % x: Parameter variable, x \in R^nx.
    % z: Space variable, z \in R^nz.
    % Convex set: C(x) = {z: A(x, z) <= 0_{nr x 1}}.
    % Static convex sets have x = zeros(0, 1).

    properties (SetAccess = immutable)
        nx
        nz
        nr
    end

    methods (Abstract)
        % Inequality constraints for convex set.
        % Inputs:
        %   x: [nx, 1] vector, Parameter variable.
        %   z: [nz, 1] vector, Space variable.
        % Outputs:
        %   A: [nr, 1] vector.
        A(obj, x, z)

        % Derivaties of constraints.
        % Inputs:
        %   x: [nx, 1] vector, Parameter variable.
        %   z: [nz, 1] vector, Space variable.
        %   y: [nr, 1] vector, Dual (\lambda) variable.
        % Outputs:
        %   A: [nr, 1] vector.
        %   dAdx: [nr, nx] matrix,
        %       x-derivative of A, D_x A(x, z).
        %   dAdz: [nr, nz] matrix,
        %       z-derivative of A, D_z A(x, z).
        %   d2Adxz_y: [nz, nx] matrix,
        %       Tensor contraction of xz-derivative of A with y,
        %       D_xz A(x, z) . y = \sum_{i=1}^{nr} y_i . D_x grad_z A_i(x, z).
        %   d2Adzz_y: [nz, nz] matrix,
        %       Tensor contraction of zz-derivative of A,
        %       D_zz A(x, z) . y = \sum_{i=1}^{nr} y_i . hess_z A_i(x, z).
        derivatives(obj, x, z, y)
    end

    methods (Access = public)
        function obj = AbstractConvexSet(nx, nz, nr)
            obj.nx = nx;
            obj.nz = nz;
            obj.nr = nr;
        end

        function [] = check_dims(obj, x_test)
            disp(['Class: ' class(obj) ' < ' ...
                strjoin(superclasses(class(obj)), ', ') ':']);
            disp(['Num of constraints: ' num2str(obj.nr)]);
            disp(['Parameter dim: [' num2str(obj.nx) ', 1]']);
            disp(['Space dim    : [' num2str(obj.nz) ', 1]']);

            if nargin < 2
                x_test = zeros(obj.nx, 1);
            end
            z_test = zeros(obj.nz, 1);
            y_test = zeros(obj.nr, 1);
            A = obj.A(x_test, z_test);
            if ~isequal(size(A), [obj.nr, 1])
                disp(['Size of obj.A is not [' num2str(obj.nr) ', 1];' ...
                    'got [' num2str(size(A)) ']']);
            end
            [~, dAdx, dAdz, d2Adxz_y, d2Adzz_y] = ...
                obj.derivatives(x_test, z_test, y_test);
            if ~isequal(size(dAdx), [obj.nr, obj.nx])
                disp(['Size of obj.dAdx is not [' num2str([obj.nr, obj.nx]) '];' ...
                    'got [' num2str(size(dAdx)) ']']);
            end
            if ~isequal(size(dAdz), [obj.nr, obj.nz])
                disp(['Size of obj.dAdz is not [' num2str([obj.nr, obj.nz]) '];' ...
                    'got [' num2str(size(dAdz)) ']']);
            end
            if ~isequal(size(d2Adxz_y), [obj.nz, obj.nx])
                disp(['Size of obj.d2Adxz_y is not [' num2str([obj.nz, obj.nx]) '];' ...
                    'got [' num2str(size(d2Adxz_y)) ']']);
            end
            if ~isequal(size(d2Adzz_y), [obj.nz, obj.nz])
                disp(['Size of obj.d2Adzz_y is not [' num2str([obj.nz, obj.nz]) '];' ...
                    'got [' num2str(size(d2Adzz_y)) ']']);
            end
        end

        function [] = check_derivatives(obj, x_test, z_test, y_test, Pi_T_x)
            % Inputs:
            %   x_test, z_test, y_test: Test points at which derivatives
            %       are evaluated.
            %   Pi_T_x: (optional) Projection matrix onto the tangent space
            %       of X (the parameter manifold) at x_test. If provided,
            %       x-derivatives are only tested on the tangent space.
            if nargin < 5
                Pi_T_x = eye(obj.nx);
            end
            if nargin < 4 || isempty(y_test)
                y_test = 10 * ones(obj.nr, 1);
            end
            if nargin < 3 || isempty(z_test)
                z_test = zeros(obj.nz, 1);
            end
            if nargin < 2 || isempty(x_test)
                x_test = zeros(obj.nx, 1);
            end

            function [f, df] = x_deriv(obj, x_test, z_test, y_test)
                [f, df, ~,~,~] = obj.derivatives(x_test, z_test, y_test);
            end
            function [f, df] = z_deriv(obj, x_test, z_test, y_test)
                [f, ~, df, ~,~] = obj.derivatives(x_test, z_test, y_test);
            end
            function [f, df] = xz_deriv(obj, x_test, z_test, y_test)
                [~,~ ,f, df, ~] = obj.derivatives(x_test, z_test, y_test);
                f = f' * y_test;
            end
            function [f, df] = zz_deriv(obj, x_test, z_test, y_test)
                [~,~ ,f, ~, df] = obj.derivatives(x_test, z_test, y_test);
                f = f' * y_test;
            end
            x_deriv_hdl = @(x) x_deriv(obj, x, z_test, y_test);
            z_deriv_hdl = @(z) z_deriv(obj, x_test, z, y_test);
            xz_deriv_hdl = @(x) xz_deriv(obj, x, z_test, y_test);
            zz_deriv_hdl = @(z) zz_deriv(obj, x_test, z, y_test);

            rng default;
            tolerance = 1e-6; % (default)
            opts = optimoptions('fmincon', FiniteDifferenceType = 'central', ...
                FiniteDifferenceStepSize = 1e-6);
            deriv_valid = true;
            [~, err] = checkGradients(x_deriv_hdl, x_test, opts, ...
                Display = 'off', Tolerance = tolerance);
            if norm(err.Objective * Pi_T_x, Inf) > tolerance
                disp('dAdx is incorrect; err: ');
                disp(num2str(err.Objective * Pi_T_x));
                deriv_valid = false;
            end
            [valid, err] = checkGradients(z_deriv_hdl, z_test, opts, ...
                Display = 'off', Tolerance = tolerance);
            if ~valid
                disp('dAdz is incorrect; err: ');
                disp(num2str(err.Objective));
                deriv_valid = false;
            end
            [~, err] = checkGradients(xz_deriv_hdl, x_test, opts, ...
                Display = 'off', Tolerance = tolerance);
            if norm(err.Objective * Pi_T_x, Inf) > tolerance
                disp('d2Adxz_y is incorrect; err: ');
                disp(num2str(err.Objective * Pi_T_x));
                deriv_valid = false;
            end
            [valid, err] = checkGradients(zz_deriv_hdl, z_test, opts, ...
                Display = 'off', Tolerance = tolerance);
            if ~valid
                disp('d2Adzz_y is incorrect; err: ');
                disp(num2str(err.Objective));
                deriv_valid = false;
            end

            if deriv_valid
                disp(['Derivatives are correct up to tolerance = ' ...
                    num2str(tolerance)]);
            end
        end
    end
end
