%% Example 1: KKT ODE.
% Example verifying the correctness of the KKT ODE and distance derivative
% for convex sets.

clear

path_ = path;
path(path_, genpath('..\..\'));


%% Define convex sets and hyper-parameters.
V1_2d = [-1, -1; -1, 1; 1, 1; 1, -1; 0, 0; 1, 1; 0, 2];
V1_3d = [V1_2d, -ones(7, 1); V1_2d, ones(7, 1)];
C1_3d = StaticPolytope(V1_3d);
C1_2d = StaticPolytope(V1_2d);
% C1_3d = SmoothPolytope(3, C1_3d, 10);
% C1_3d.radius = 5;
% C1_3d.center = zeros(3, 1);

Q_3d = [3, 1, 0; 1, 2, 0; 0, 0, 1];
Q_2d = Q_3d(1:2, 1:2);
C2_2d = RigidEllipsoid(2, Q_2d);
C2_3d = FlexibleEllipsoid(Q_3d);


%% Define state sequences.
dt = 0.01;
T = 25;
t_seq = 0:dt:T;
nT = length(t_seq);

% SmoothPolytope is static.
R1 = eye(3);
x_seq1_2d = zeros(0, nT);
x_seq1_3d = zeros(0, nT);
% x_seq1_3d = [zeros(3, 1); R1(:)] * ones(1, nT);
% RigidEllipsoid revolves in x-y, oscillates in z, and rotates in z.
p_seq2 = [7 * cos(2 * pi * t_seq / 10);
    7 * sin(2 * pi * t_seq / 10);
    3 * sin(2 * pi * t_seq / 11)];
rot_z_seq2 = 2 * pi * t_seq / 9;
R_seq2 = [cos(rot_z_seq2); sin(rot_z_seq2); zeros(1, nT); -sin(rot_z_seq2); ...
    cos(rot_z_seq2); zeros(3, nT); ones(1, nT)];
x_seq2_3d = [p_seq2; R_seq2];
x_seq2_2d = x_seq2_3d([1, 2, 4, 5, 7, 8], :);


%% Minimum distance ODE test between StaticPolytope and SmoothPolytope.

out_2d = compare_opt_ode(t_seq, x_seq1_2d, C1_2d, x_seq2_2d, C2_2d);
disp('2D case: StaticPolytope - dynamic RigidEllipsoid:');
print_stats(out_2d);
disp(' ');

out_3d = compare_opt_ode(t_seq, x_seq1_3d, C1_3d, x_seq2_3d, C2_3d);
disp('3D case: Static SmoothPolytope - dynamic FlexibleEllipsoid:');
print_stats(out_3d);
disp(' ');
Tf = round(20 / dt);
timestamps = 1 + round([2.5, 6., 7.5] / dt);

% NOTE: Change figure size to [370, 250] points.
til = plot_figure(out_3d, x_seq1_3d(:, 1), C1_3d, x_seq2_3d, C2_3d, Tf, timestamps);
save_plot_as_one = false;
if false
    if save_plot_as_one
        fig = gcf;
        print(fig, strcat('../../fig/kkt-error.png'), '-dpng', '-r1000');
        print(fig, strcat('../../fig/kkt-error.eps'), '-depsc', '-r600');
    else
        child_ = flipud(til.Children);
        child_ = child_(1:2:end);
        title = '../../fig/kkt-error-';
        subtitles = {'env', 'kkt', 'dist', 'Ddist'};
        for i = 1:4
            exportgraphics(child_(i), strcat(title, subtitles{i}, '.png'), ...
                Resolution = 1000);
            exportgraphics(child_(i), strcat(title, subtitles{i}, '.eps'), ...
                Resolution = 600);
        end
    end
end

%%
% figure()
% hold on
% plot(t_seq(1:end-1), out_3d.Ddist2_opt_seq, '-r');
% % plot(t_seq(1:end-1), out_3d.Ddist2_ode_seq, '-g');
% plot(t_seq(1:end-1), out_3d.Ddist2_analytic_seq, '-b');
%
% % figure()
% % plot(t_seq, out_3d.kkt_err_seq)
% % set(gca, 'YScale', 'log')
%
% figure()
% hold on
% % plot(t_seq(1:end-1), out_3d.Ddist2_opt_seq - out_3d.Ddist2_ode_seq, '-r');
% plot(t_seq(1:end-1), zeros(1, nT-1), '--k');
% plot(t_seq(1:end-1), out_3d.Ddist2_opt_seq ./ (2*sqrt(out_3d.dist2_opt_seq(1:end-1))) - out_3d.Ddist2_analytic_seq ./ (2*sqrt(out_3d.dist2_ode_seq(1:end-1))), '-g');
% % plot(t_seq(1:end-1), out_3d.Ddist2_ode_seq - out_3d.Ddist2_analytic_seq, '-b');
%
% norm(out_3d.Ddist2_opt_seq - out_3d.Ddist2_analytic_seq, Inf)
% mean(out_3d.Ddist2_opt_seq - out_3d.Ddist2_analytic_seq)


%%
path(path_);
