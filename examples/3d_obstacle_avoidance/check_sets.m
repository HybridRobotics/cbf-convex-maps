path(path, genpath('..\kkt_ode\'));

rng(1,"twister");

sets = cell(2, 1);
for i = 1:2
    nrandpts = randi([10, 15]); % [7, 9].
    j = 0;
    V_ = zeros(nrandpts, 3);
    while true
        s2_pt = 2 * rand(1, 3) - ones(1, 3);
        if norm(s2_pt, 2) < 1e-3
            continue;
        end
        j = j + 1;
        s2_pt = s2_pt / norm(s2_pt, 2);
        rad = (1.25 - 0.5) * rand() + 0.5;
        V_(j,:) = rad * s2_pt;
        if j == nrandpts
            break
        end
    end
    P_ = Polyhedron(V_);
    P_.minHRep();
    P.A_mat = P_.A;
    P.b_vec = P_.b;
    set = SmoothPolytope4D(P, 1, 5);
    set.radius = 5;
    set.center = [0; 0; 0];

    R_ = eye(3);
    x_ = [0; 0; 0; 1; 1; 1; R_(:)];
    set.check_derivatives(x_, [0; 0; 0; 0], 1);

    sets{i} = set;
end


t_seq = 0:0.01:25;
nT = length(t_seq);

R_ = eye(3);
x_seq1 = [zeros(6, 1); R_(:)] * ones(1, nT);
r_seq2 = 15 + sin(2*pi/5 * t_seq);
p_seq2 = [r_seq2 .* cos(2*pi/25 * t_seq);
    r_seq2 .* sin(2*pi/25 * t_seq);
    sin(2*pi/10 * t_seq)];
v_seq2 = [-2*pi/25 * (r_seq2 .* sin(2*pi/25 * t_seq)) + 2*pi/5 * (cos(2*pi/5 * t_seq) .* cos(2*pi/25 * t_seq));
    2*pi/25 * (r_seq2 .* cos(2*pi/25 * t_seq)) + 2*pi/5 * (cos(2*pi/5 * t_seq) .* sin(2*pi/25 * t_seq));
    2*pi/10 * cos(2*pi/10 * t_seq)];
x_seq2 = [p_seq2; v_seq2; R_(:) * ones(1, nT)];

out = compare_opt_ode(t_seq, x_seq1, C1, x_seq2, C2);
print_stats(out);
