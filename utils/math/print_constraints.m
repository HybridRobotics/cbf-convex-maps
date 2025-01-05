function [] = print_constraints(A, l, u, sep_line_width)
    % Prints the linear constraints l <= Ax <= u in the form:
    % l_i | A_i | u_i, for each row of A.
    % A can be a sparse matrix.
    A_d = full(A);
    n_rows = size(A_d, 1);

    if isempty(l)
        l = -Inf * ones(n_rows, 1);
    end
    if isempty(u)
        u = Inf * ones(n_rows, 1);
    end

    assert (isequal(size(l), [n_rows, 1]));
    assert (isequal(size(u), [n_rows, 1]));

    sep_mat = char(repmat(' | ', n_rows, 1));
    A_str = num2str(A_d, '%6.2f');
    l_str = num2str(l);
    u_str = num2str(u);
    ineq_str = [l_str, sep_mat, A_str, sep_mat, u_str];

    if nargin < 4
        sep_line_width = size(ineq_str, 2);
    end
    disp(char(repmat('-', 1, sep_line_width)));
    disp('Linear inequality constraints:');
    disp(ineq_str);
    disp(char(repmat('-', 1, sep_line_width)));
end
