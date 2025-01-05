function [pctl_values] = pctl(arr, p)
    % Computes different p-percentile values for the array arr.
    % p: Array of percentile values between 0-100.
    l = length(arr);
    p_seq = linspace(0.5/l, 1 - 0.5/l, l);
    func = @(p) interp1(p_seq, sort(arr), p * 0.01, 'spline');

    pctl_values = zeros(size(p));
    for i = 1:length(p)
        pctl_values(i) = func(p(i));
    end
end
