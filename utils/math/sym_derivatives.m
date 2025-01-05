function [] = sym_derivatives(A, x, z, params, file_path)
    % Computes derivatives from symbolic function.
    %
    % Inputs:
    %   A: Symbolic real-valued function of x and z.
    %   x: State symbolic variable.
    %   z: Space symbolic variable.
    %   params: Parameters to A.
    %   file_path: File path to saved function.
    dAdx = gradient(A, x)';
    dAdz = gradient(A, z)';
    d2Adxz = jacobian(dAdz, x);
    d2Adzz = jacobian(dAdz, z);
    matlabFunction(A, dAdx, dAdz, d2Adxz, d2Adzz, File = file_path, ...
    Outputs = {'A', 'dAdx', 'dAdz', 'd2Adxz', 'd2Adzz'}, ...
    Vars = {x, z, params}, Optimize = true);
    rehash
end
