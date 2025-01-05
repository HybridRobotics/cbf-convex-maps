path_ = path;
path(path_, genpath('..\..\'));

syms z [2 1] real % (x, y) coordinates.
syms x [3 1] real % (x, y, \theta).

rot = [cos(x(3)), -sin(x(3)); sin(x(3)), cos(x(3))];
z_inv = rot' * (z - x(1:2));

folder_path = strcat(fileparts(mfilename('fullpath')), ...
    '\generated_functions\');


% Offset circle.
syms p [2 1] real
syms R real
offset_circ = (z_inv(1) - p(1))^2 + (z_inv(2) - p(2))^2 - R^2;
file_path = strcat(folder_path, 'offset_circle');
if ~isfile(file_path)
    sym_derivatives(offset_circ, x, z, [p; R], file_path);
end

% 4-norm level set.
syms lev real
norm4_lev = z_inv(1)^4 + z_inv(2)^4 - lev;
file_path = strcat(folder_path, 'norm4_lev');
if ~isfile(file_path)
    sym_derivatives(norm4_lev, x, z, lev, file_path);
end

%% Test Classes.
% OffsetCircle3.
p_mat = [0.1, 0.1; -0.5, 0; 0, -0.2];
R_arr = [1; 1; 1];
C1 = OffsetCircles3(p_mat, R_arr);
C1.check_dims();
C1.check_derivatives();
C1.radius = 5;
C1.center = [0; 0];
figure();
ax = gca;
C1.plot_surf(zeros(3, 1), ax, 'r', 1, 1);

%
lev = 2;
C2 = Norm4Level(lev);
C2.check_dims();
C2.check_derivatives();
C2.radius = 5;
C2.center = [0; 0];
figure();
ax = gca;
C2.plot_surf(zeros(3, 1), ax, 'r', 1, 1);
