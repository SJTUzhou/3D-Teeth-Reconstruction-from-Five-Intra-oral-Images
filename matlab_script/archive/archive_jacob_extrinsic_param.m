function func_jacob_extrinsic_param = jacob_extrinsic_param(f, dpix, u0, v0, X_corre_pred, X_corre_pred_normals, P_true, var_point, var_plane)
    syms rx ry rz tx ty tz;
%     X_corre_pred = sym('X_corre_pred', [numPixel 3]);
%     X_corre_pred_normals = sym('X_corre_pred_normals', [numPixel 3]);
%     P_true = sym('P_true', [numPixel 2]);
    rotX = [1,      0,       0;
            0, cos(rx), sin(rx); 
            0, -sin(rx), cos(rx)];
        
    rotY = [cos(ry),  0,  -sin(ry);
            0,  1,       0;
            sin(ry),  0,  cos(ry)]; 
    rotZ = [cos(rz),  sin(rz),  0;
           -sin(rz),  cos(rz),  0;
           0,       0,   1];
       
    intrMat1 = [f, 0, 0;
                0, f, 0;
                0, 0, 1];
            
    intrMat2 = [1./dpix, 0., 0.;
                0., 1./dpix, 0.;
                u0, v0, 1.];

    extrViewRotMat = rotX * rotY * rotZ;
    intrProjMat = intrMat1 * intrMat2;
    
    X_cam_corre_pred = X_corre_pred * extrViewRotMat + [tx ty tz];
    X_cam_corre_pred_normals = X_corre_pred_normals * extrViewRotMat;
%     obj_dist = 10;
    P_corre_pred = (X_cam_corre_pred ./ X_cam_corre_pred(:,3)) * intrProjMat;
%     P_corre_pred = (X_cam_corre_pred ./obj_dist) * intrProjMat;
    P_corre_pred = P_corre_pred(:,1:2);
    fprintf('size(P_corre_pred) is %s\n', mat2str(size(P_corre_pred)))
    P_corre_pred_normals = X_cam_corre_pred_normals(:,1:2);
    fprintf('size(P_corre_pred_normals) is %s\n', mat2str(size(P_corre_pred_normals)))
    P_corre_pred_normals = P_corre_pred_normals ./ sum(P_corre_pred_normals.^2, 2); % dim=2: norm for each row
    
    P_error = P_corre_pred - P_true;
    loss_point = sum(sum(P_error.^2, 2)) / var_point;
    loss_plane = sum(sum(P_error.*P_corre_pred_normals, 2).^2) / var_plane;
    loss = loss_point + loss_plane;
%     grad_vec = [simplify(diff(loss, tx, 1)), simplify(diff(loss, ty, 1)), simplify(diff(loss, tz, 1)), ... 
%                 simplify(diff(loss, rx, 1)), simplify(diff(loss, ry, 1)), simplify(diff(loss, rz, 1))];
    grad_vec = [diff(loss, tx, 1), diff(loss, ty, 1), diff(loss, tz, 1), ... 
                diff(loss, rx, 1), diff(loss, ry, 1), diff(loss, rz, 1)];
    fprintf('size(grad_vec) is %s\n', mat2str(size(grad_vec)))
    
    func_jacob_extrinsic_param = matlabFunction(grad_vec);
end

