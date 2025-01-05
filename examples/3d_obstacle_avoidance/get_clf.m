function [func] = get_clf(type)
    switch type
        case 'double-integrator'
            func = @(x, xd) soubleintegrator(x, xd);
        case 'simple-quadrotor'
            return
    end
end

function [V, DV] = doubleintegrator_clf(x, xd)
    k_R = 4;
    k_v = 2;

    p = x(1:3);
    v = x(4:6);
    R = reshape(x(7:end), 3, 3);
    pd = xd(1:3);
    vd = xd(4:6);
    Rd = reshape(xd(7:end), 3, 3);

    V = 1/2 * sum((p - pd).^2) + 1/2 * k_v * sum((v - vd).^2) + ...
        k_R/2 * trace(eye(2) - Rd' * R);
    DV = [(p - pd)', k_v * (v - vd)', -k_R/2 * Rd(:)'];
end

function [V, DV] = simplequadrotor_clf(x, xd)
    %
end
