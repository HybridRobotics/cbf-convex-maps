clear

current_path = path;
path(path, genpath('..\'));


%% StaticPolytope test.
sp = [];
% Generate from vertices and check redundancy.
V = [-1, -1; -1, 1; 1, 1; 1, -1; 0, 0; 1, 1];
V = V + ones(size(V, 1), 1) * [5, 5];
sp = [sp, StaticPolytope(V)];
sp(end).check_dims();
display_static_polytope(sp(end));
% Generate from h-rep, without reduction.
A = [1, 0; 0, 1; 1, 1];
b = [1; 1; 2];
sp = [sp, StaticPolytope(A, b)];
sp(end).check_dims();
display_static_polytope(sp(end));
% Generate from h-rep, with reduction.
sp = [sp, StaticPolytope(A, b, true)];
sp(end).check_dims();
sp(end).check_derivatives();
display_static_polytope(sp(end));


%% RigidEllipsoid test.
re = [];
Q = [3, 1, 0; 1, 2, 0; 0, 0, 1];
re = [re, RigidEllipsoid(3, Q)];
re(end).check_dims();
theta_z = 15 * pi/180;
R = [cos(theta_z), -sin(theta_z), 0; sin(theta_z), cos(theta_z), 0; 0, 0, 1];
x_test = [ones(3, 1); reshape(R, 9, 1)];
Pi_T_x = proj_se3(R);
re(end).check_derivatives(x_test, [], []);
disp(' ');


%% SmoothPolytope test.
smp = [];
V = [-1, -1; -1, 1; 1, 1; 1, -1; 0, 0; 1, 1; 0, 2];
V = [V, -ones(7, 1); V, ones(7, 1)];
V = V + ones(size(V, 1), 1) * [5, 5, 5];
poly = StaticPolytope(V);
smp = [smp, SmoothPolytope(3, poly, 10)];
smp(end).check_dims();
smp(end).check_derivatives();
disp(' ');


%%
path(current_path);


%% Polytope display function.
function [] = display_static_polytope(p)
    if ~isempty(p.P)
        display(p.P);
    else
        disp('[]');
    end
    disp(['nr: ' num2str(p.nr) ', nz: ' num2str(p.nz)]);
    print_constraints(p.A_mat, [], p.b_vec);
    disp(' ');
end

%% Projection matrix for (\dot{p}, \dot{R}) onto T_{(p,R)}.
% SE(3)\dot{R} is projected as (\dot{R} - R \dot{R}^T R)/2.
function [Pi_T_x] = proj_se3(Rot)
    R = sym('R', [9 1]);
    R = reshape(R, [3 3]);
    V = sym('V', [9 1]);
    V = reshape(V, [3 3]);
    sym_proj = 1/2 * (V - R * V' * R);
    sym_proj_vectorized = jacobian(sym_proj(:), V(:));
    proj_vectorized = double(subs(sym_proj_vectorized, R, Rot));

    Pi_T_x = blkdiag(eye(3), proj_vectorized);
end
