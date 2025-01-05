classdef Robot < handle
    % Robot structure containing dynamical system and convex sets.

    properties (SetAccess = public)
        % DynamicalSystem object.
        system

        nx
        nu
        nz

        % Cell of AbstractConvexSet objects.
        sets
        nsets
    end

    methods
        function obj = Robot(system, sets)
            obj.system = system;
            obj.sets = sets;
            obj.nsets = numel(sets);
            % Check dimensions.
            obj.nx = system.nx;
            obj.nu = system.nu;
            obj.nz = sets{1}.nz;
            for i = 1:obj.nsets
                if obj.nx ~= sets{i}.nx || obj.nz ~= sets{i}.nz
                    error('State/Space dimensions do not match');
                end
            end
        end
    end
end
