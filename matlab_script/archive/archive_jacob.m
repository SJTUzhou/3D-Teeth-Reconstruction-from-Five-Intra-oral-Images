%%
clear;
numPixel = 100;
X_corre_pred = rand(numPixel,3);
X_corre_pred_normals = rand(numPixel,3);
P_true = rand(numPixel,2);
f = 105;
dpix = 0.06;
u0 = 400;
v0 = 300;
var_point = 500;
var_plane = 1000;
func_jacob_extrinsic_param = jacob_extrinsic_param(f, dpix, u0, v0, X_corre_pred, X_corre_pred_normals, ...
    P_true, var_point, var_plane);
% func_jacob_extrinsic_param = jacob_extrinsic_param(f, dpix, u0, v0, numPixel, var_point, var_plane);
% grad_vec = jacob_extrinsic_param(f, dpix, u0, v0, X_corre_pred, X_corre_pred_normals, P_true, var_point, var_plane);
%%
% syms rx ry rz tx ty tz;
% vpa(subs(grad_vec, {rx, ry, rz, tx, ty, tz}, {1,1,1,1,1,1}))