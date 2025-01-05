classdef StaticPolytope < AbstractConvexSet
    % Bounded polyhedron set with no parameters.
    %
    % P = {z: A_mat z <= b_vec}.

    properties (SetAccess = immutable)
        A_mat
        b_vec
    end

    properties (SetAccess = private)
        P = [] % Polyhedron struct.
        V = [] % [-1, nz] matrix, Vertices of polytope.
    end

    properties (Access = public)
        surf_pts = []
    end

    methods
        function obj = StaticPolytope(A, b, reduce)
            % Inputs:
            %   Case 1: nargin == 1
            %       A: [-1, nz] matrix, V-rep of polytope.
            %   Case 2: nargin > 1
            %       A, b: H-rep of a polytope.
            %       reduce: whether to reduce H-rep or not (default).
            if nargin == 1
                % Set Polytope from vertex set A (=V).
                P = Polyhedron(A);
                P.minHRep();
                V = P.V;
                A_mat = P.A;
                b_vec = P.b;
            elseif nargin == 2 || ~reduce
                % Set Polytope from halfspaces Az <= b.
                P = [];
                V = [];
                A_mat = A;
                b_vec = b;
            else
                % Set Polytope from halfspaces Az <= b and remove redundant
                % constraints.
                P = Polyhedron(A, b);
                P.minHRep();
                V = [];
                A_mat = P.A;
                b_vec = P.b;
            end

            assert(SetUtils.is_polytope_solid(A_mat, b_vec), ...
                'Polyhedron is not solid');

            obj = obj@AbstractConvexSet(0, size(A_mat, 2), size(A_mat, 1));
            obj.P = P;
            obj.V = V;
            obj.A_mat = A_mat;
            obj.b_vec = b_vec;
        end

        function cons = A(obj, ~, z)
            cons = obj.A_mat * z - obj.b_vec;
        end

        function [A, dAdx, dAdz, d2Adxz_y, d2Adzz_y] = derivatives(obj, ~, z, ~)
            A = obj.A_mat * z - obj.b_vec;
            dAdx = zeros(obj.nr, 0);
            dAdz = obj.A_mat;
            d2Adxz_y = zeros(obj.nz, 0);
            d2Adzz_y = zeros(obj.nz);
        end

        function [obj] = plot_surf(obj, ~, hdl, fc, fa, ~)
            if obj.nz == 2
                if isempty(obj.surf_pts)
                    P_ = Polyhedron(obj.A_mat, obj.b_vec);
                    P_.minHRep();
                    V_ = P_.V;
                    angles = atan2( V_(:, 2) - mean(V_(:, 2)), ...
                        V_(:, 1) - mean(V_(:, 1)) );
                    [~, idx] = sort(angles);
                    obj.surf_pts = V_(idx, :);
                end
                fill(hdl, obj.surf_pts(:, 1), obj.surf_pts(:, 2), fc, ...
                    'FaceAlpha', fa, 'EdgeColor', [0, 0, 0]);
            elseif obj.nz == 3
                if isempty(obj.surf_pts)
                    P_ = Polyhedron(obj.A_mat, obj.b_vec);
                    P_.minHRep();
                    V_ = P_.V;
                    obj.V = P_.V;
                    obj.surf_pts = convhull(V_(:, 1), V_(:, 2), V_(:, 3), ...
                        'Simplify', true);
                end
                trisurf(obj.surf_pts, obj.V(:, 1), obj.V(:, 2), obj.V(:, 3), ...
                    'FaceColor', fc, 'FaceAlpha', fa, ...
                    'EdgeColor', [0, 0, 0], 'Parent', hdl);
            else
                error('Plotting for nz not 2 or 3 is not supported');
            end
        end
    end
end
