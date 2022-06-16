classdef EmOptMStep < handle
    properties
        % variable fixed from previous Expectation Step launched by Python
        X_Mu; % array (3,1500,numAllTeeth)
        X_Mu_centroids; % array (3,numAllTeeth)
        X_Mu_pred; % cell (1,5) 每张图片中涉及的均值点云模型中的点坐标
        X_Mu_pred_normals; % cell (1,5) 每张图片中涉及的均值点云模型中的点的法向量
        visIdx;  % cell (1,5) 每张图片中涉及的牙齿的index; remark: [+1 from np idx]
        corre_pred_idx; % cell (1,5); remark: [+1 from np idx]
        ul_sp; %Map, 在左侧照，右侧照，正面照中上下牙id分界; remark: [+1 from np idx]
        P_true; % cell (1,5) 每张图片edge pixel坐标
        P_true_weights; % cell (1,5) 每张图片edge pixel权重
        M; % array (1,5) 每张图片edge pixel的数量
        NumAllTeeth; % int, 患者牙齿数量约为28
        
        % parameters optimized in Stage0,1,2
        ex_rxyz; % array(5,3)
        ex_txyz; % array(3,1)
        focLth; % array(1,5)
        dpix; % array(1,5)
        u0; % array(1,5)
        v0; % array(1,5)
        rela_rxyz; % array(1,3)
        rela_txyz; % array(3,1)

        % parameters optimized in Stage1
        rowScaleXZ; % array(1,2)

        % parameters optimized in Stage2 
        scales; % array (1,numAllTeeth)
        rotAngleXYZs; % array (numAllTeeth,3)
        transVecXYZs; % array (3,numAllTeeth)
        scaleStd; % double
        rotAngleStd; % double
        transVecStd; % double

        % Constant variables (may be loaded from data)
        varPoint; % double
        varPlane; % double
        weightAniScale; % double
        varScale; % double; == scaleStd^2
        weightTeethPose;
        invCovMatOfScale; % array (numAllTeeth, numAllTeeth)
        invCovMatOfPose; % array (6,6,numAllTeeth)
        weightPhotos;
        photoTypes; % array (1,5) dataType:PHOTO
        
        
    end

    methods
        % 构造函数
        function obj = EmOptMStep(np_X_Mu, np_X_Mu_pred, np_X_Mu_pred_normals,...
                np_visIdx, np_corre_pred_idx, np_P_true, np_P_true_weights, np_ex_rxyz, np_ex_txyz, np_focLth, np_dpix,...
                np_u0, np_v0, np_rela_rxyz, np_rela_txyz, np_rowScaleXZ, np_scales,...
                np_rotAngleXYZs, np_transVecXYZs, np_invCovMatOfScale, np_invCovMatOfPose)

            obj.X_Mu = double(permute(np_X_Mu,[3,2,1])); %transpose
            [~,~,obj.NumAllTeeth] = size(obj.X_Mu);
            obj.X_Mu_centroids = squeeze(mean(obj.X_Mu, 2));

            obj.visIdx = cell(1,5);
            diffNumTeethFlag = isa(np_visIdx, "cell");
            for ph = 1:5
                if diffNumTeethFlag == true
                    obj.visIdx{ph} = 1 + np_visIdx{ph}; % remark: [+1 from np idx]
                else
                    obj.visIdx{ph} = 1 + np_visIdx(ph,:);
                end
            end

            obj.X_Mu_pred = cell(1,5);
            for ph = 1:5 % follow fixed photo order
                obj.X_Mu_pred{ph} = {};
                if diffNumTeethFlag == true
                    np_X_Mu_pred_ph = np_X_Mu_pred{ph};
                else
                    np_X_Mu_pred_ph = np_X_Mu_pred(ph,:);
                end
                for i = 1:length(np_X_Mu_pred_ph)
                    obj.X_Mu_pred{ph}{i} = double(np_X_Mu_pred_ph{i}');
                end
            end
            
            obj.X_Mu_pred_normals = cell(1,5);
            for ph = 1:5
                obj.X_Mu_pred_normals{ph} = {};
                if diffNumTeethFlag == true
                    np_X_Mu_pred_normals_ph = np_X_Mu_pred_normals{ph};
                else
                    np_X_Mu_pred_normals_ph = np_X_Mu_pred_normals(ph,:);
                end
                for i = 1:length(np_X_Mu_pred_normals_ph)
                    obj.X_Mu_pred_normals{ph}{i} = double(np_X_Mu_pred_normals_ph{i}');
                end
            end

            
            
            obj.corre_pred_idx = cell(1,5);
            for ph = 1:5
                obj.corre_pred_idx{ph} = 1 + np_corre_pred_idx{ph}; % remark: [+1 from np idx]
            end
            
            numUpperTooth = length(obj.visIdx{uint32(PHOTO.UPPER)});
            obj.ul_sp = containers.Map([uint32(PHOTO.LEFT), uint32(PHOTO.RIGHT), uint32(PHOTO.FRONTAL)], [0,0,0]);
            for phType = [PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]
                ph = uint32(phType);
                obj.ul_sp(ph) = find(obj.visIdx{ph}>numUpperTooth, 1);
            end
            
            obj.P_true = cell(1,5);
            obj.P_true_weights = cell(1,5);
            for ph = 1:5
                obj.P_true{ph} = double(np_P_true{ph}'); %transpose
                obj.P_true_weights{ph} = double(np_P_true_weights{ph}');
            end

            obj.M = zeros(1,5);
            for ph = 1:5
                [~,obj.M(ph)] = size(obj.P_true{ph}); %每张图片edge pixel的数量
            end
            
            % parameters optimized in Stage1
            obj.ex_rxyz = double(np_ex_rxyz);
            obj.ex_txyz = double(np_ex_txyz'); %transpose
            obj.focLth = double(np_focLth);
            obj.dpix = double(np_dpix);
            obj.u0 = double(np_u0);
            obj.v0 = double(np_v0);
            obj.rela_rxyz = double(np_rela_rxyz);
            obj.rela_txyz = double(np_rela_txyz'); %transpose
            obj.rowScaleXZ = double(np_rowScaleXZ);
            
            % parameters optimized in Stage2
            obj.scales = double(np_scales); 
            obj.rotAngleXYZs = double(np_rotAngleXYZs);
            obj.transVecXYZs = double(np_transVecXYZs'); %transpose

            obj.scaleStd = 0.0678;
            obj.rotAngleStd = 1.1732;
            obj.transVecStd = 0.1416;

            
            obj.varPoint = 25;
            obj.varPlane = 0.5;
            obj.weightAniScale = 1;
            obj.varScale = obj.scaleStd^2;
            obj.weightTeethPose = 1;
            obj.invCovMatOfScale = double(np_invCovMatOfScale); % symmetrical
            obj.invCovMatOfPose = permute(double(np_invCovMatOfPose), [3,2,1]); %transpose
            obj.weightPhotos = [3, 3, 1, 1, 1];
            obj.photoTypes = [PHOTO.UPPER, PHOTO.LOWER, PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL];


        end
        
        % 计算2D像素空间中的损失
        function pixelError = computePixelResidualError(obj, photoType, scales, rotAngleXYZs, transVecXYZs,...
            ex_R, ex_txyz, intrProjMat, rela_R, rela_txyz, rowScaleXZ, stage)
            % scales.shape = (1,numToothPh)
            % rotAngleXYZs.shape = (numToothPh,3)
            % transVecXYZs.shape = (3,numToothPh)
            % ex_txyz.shape = (3,1)
            % rela_txyz.shape = (3,1)

            ph = uint32(photoType); % photoType: PHOTO enumeration 1,2,3,4,5
            tIdx = obj.visIdx{ph};
            numToothPh = length(tIdx); %本张照片中对应的牙齿数量

            X_deformed_pred = obj.X_Mu_pred{ph}; % cell (1,numToothPh)
            X_deformed_pred_normals = obj.X_Mu_pred_normals{ph}; % cell (1,numToothPh)

            X_trans_pred = obj.X_Mu_pred{ph}; % cell (1,numToothPh)
            X_trans_pred_normals = obj.X_Mu_pred_normals{ph}; % cell (1,numToothPh)
               
            %考虑 deformation in shape subspace
            if stage >= 3 
            end

            %考虑每颗牙齿的相对位姿和尺寸
            if stage >= 2 
                rotMats = EmOptMStep.computeRotMats(rotAngleXYZs);
                for i = 1:numToothPh
                    x = X_deformed_pred{i}; % array (3,?)
                    if isempty(x) % 某颗牙齿的轮廓被完全遮住的情况
                        continue;
                    end
                    xn = X_deformed_pred_normals{i}; % array (3,?)
                    xc = obj.X_Mu_centroids(:,i); % array (3,1); 均值模型中这颗牙齿的重心（坐标原点）
                    X_trans_pred{i} = scales(i) * rotMats(:,:,i) * (x-xc) + xc + transVecXYZs(:,i);
                    X_trans_pred_normals{i} = rotMats(:,:,i) * xn;
                end
            end

            % 需要考虑上下牙列位置关系，对下牙列的点进行相对旋转和平移
            if photoType==PHOTO.LEFT || photoType==PHOTO.RIGHT || photoType==PHOTO.FRONTAL
                id_lt = obj.ul_sp(ph); % starting index of lower tooth row
                for i = id_lt:numToothPh
                    if isempty(X_trans_pred{i}) % 某颗牙齿的轮廓被完全遮住的情况
                        continue;
                    end
                    X_trans_pred{i} = rela_R * X_trans_pred{i} + rela_txyz;
                    X_trans_pred_normals{i} = rela_R * X_trans_pred_normals{i};
                end
            end

            % 根据3d-2d对应关系选择对应点并拼接
            X_pred = horzcat(X_trans_pred{:});
            X_pred_normals = horzcat(X_trans_pred_normals{:});
            X_corre_pred = X_pred(:,obj.corre_pred_idx{ph});
            X_corre_pred_normals = X_pred_normals(:,obj.corre_pred_idx{ph});
            if stage == 1 % 在优化相机参数同时，优化牙列的Anistropic scales
                X_corre_pred = [rowScaleXZ(1);1;rowScaleXZ(2)] .* X_corre_pred;
            end

            % 相机坐标系下对应点坐标和法向量
            X_cam_corre_pred = ex_R * X_corre_pred + ex_txyz; 
            X_cam_corre_pred_normals = ex_R * X_corre_pred_normals;


            % 像素坐标系下对应点坐标和法向量
            % The following assertion should be converted to non-linear constraints
            % assert(all(X_cam_corre_pred(3,:)>0), "Z-value of points should be positive"); 
            X_image = intrProjMat * (X_cam_corre_pred./X_cam_corre_pred(3,:));
            P_corre_pred = X_image([1,2],:);
            P_corre_pred_normals = X_cam_corre_pred_normals([1,2],:);
            P_corre_pred_normals = P_corre_pred_normals ./ vecnorm(P_corre_pred_normals); % 法向量模长为1
            % 损失函数
            errorVecUV = obj.P_true{ph} - P_corre_pred;
            resPointError = sum(obj.P_true_weights{ph} .* sum(errorVecUV.^2, 1)) ./ obj.varPoint;
            resPlaneError = sum(obj.P_true_weights{ph} .* sum(errorVecUV.*P_corre_pred_normals, 1).^2) ./ obj.varPlane;
            pixelError = (resPointError + resPlaneError) / obj.M(ph);
        end

        % 计算牙齿相对位姿的损失
        function teethPoseError = computeTeethPoseResidualError(obj, scales, rotAngleXYZs, transVecXYZs, tIdx)
            % Suppose mean(scales) = 1; mean(rotAngleXYZs) = 0; mean(transVecXYZs) = 0
            pose6Dparams = vertcat(transVecXYZs, rotAngleXYZs'); % size=(6,numToothPh)
            pose6DInvCovMat = obj.invCovMatOfPose(:,:,tIdx); % size=(6,6,numToothPh)
            scaleInvCovMat = obj.invCovMatOfScale(tIdx,tIdx); % size=(numToothPh,numToothPh)
            errorScale = (scales-1) * scaleInvCovMat * (scales'-1);
            errorPose6D = 0;
            for i =1:length(tIdx)
                errorPose6D = errorPose6D + pose6Dparams(:,i)' * pose6DInvCovMat(:,:,i) * pose6Dparams(:,i);
            end
            teethPoseError = obj.weightTeethPose * (errorScale + errorPose6D);
        end



        % Loss function of Maximization Step Stage
        function loss = MStepLoss(obj, params, pIdx, stage, step, verbose)
            % global params
            [ex_rxyz, ex_txyz, focLth, dpix, u0, v0, rela_rxyz, rela_txyz] = EmOptMStep.parseGlobalParamsOf5Views(params, pIdx);  %#ok<*PROPLC> 
            ex_R = EmOptMStep.computeRotMats(ex_rxyz); % extrinsic rotation matrix, shape (3,3,5)
            rela_R = EmOptMStep.computeRotMats(rela_rxyz); % tooth row relative rotation matrix
            intrProjMat = zeros(3,3,5);
            for ph = 1:5
                intrProjMat(:,:,ph) = EmOptMStep.IntrinsicProjectionMatrix(focLth(ph), dpix(ph), u0(ph), v0(ph)); 
            end
            rowScaleXZ = [1, 1];
            scales = obj.scales;
            rotAngleXYZs = obj.rotAngleXYZs;
            transVecXYZs = obj.transVecXYZs;
            
            % Only for stage 1
            aniScaleError = 0;
            if stage == 1
                rowScaleXZ = params(pIdx("rowScaleXZ"):pIdx("rowScaleXZ")+1);
%                 rowScaleXYZ = [rowScaleXZ(1);1;rowScaleXZ(2)];
%                 scales = prod(rowScaleXYZ)^(1/3) * ones([1,obj.NumAllTeeth]);
%                 rotAngleXYZs = zeros([obj.NumAllTeeth,3]);
%                 transVecXYZs = obj.X_Mu_centroids .* (rowScaleXYZ - 1);
%                 equiva_tIdx = 1:obj.NumAllTeeth;
%                 aniScaleError = obj.computeTeethPoseResidualError(scales,...
%                     rotAngleXYZs, transVecXYZs, equiva_tIdx);
                equiva_scale = prod(rowScaleXZ)^(1/3);
                aniScaleError = obj.weightAniScale * (equiva_scale-1)^2 / obj.varScale;
            end

            % Only for stage == 2
            if stage == 2
                % teeth pose params
                switch step
                    case 1
                        transVecXYZs = reshape(params(pIdx("tXYZs"):pIdx("tXYZs")+3*obj.NumAllTeeth-1),...
                            [3,obj.NumAllTeeth]);
                    case 2
                        rotAngleXYZs = reshape(params(pIdx("rXYZs"):pIdx("rXYZs")+3*obj.NumAllTeeth-1),...
                            [obj.NumAllTeeth,3]);
                    case 3
                        scales = params(pIdx("scales"):pIdx("scales")+obj.NumAllTeeth-1);
                    case 4
                        [scales, rotAngleXYZs, transVecXYZs] =...
                            EmOptMStep.parseTeethPoseParams(params, pIdx, obj.NumAllTeeth);
                end
            end
            % Compute M-Step Loss
            losses = zeros(1,5);
            for phType = obj.photoTypes
                ph = uint32(phType);
                tIdx = obj.visIdx{ph};
                pixelError = obj.computePixelResidualError(phType, scales(tIdx), rotAngleXYZs(tIdx,:),...
                    transVecXYZs(:,tIdx), ex_R(:,:,ph), ex_txyz(:,ph), intrProjMat(:,:,ph),...
                    rela_R, rela_txyz, rowScaleXZ, stage);
                % For Stage 2 and 3
                teethPoseError = 0;
                if stage == 2 
                    teethPoseError = obj.computeTeethPoseResidualError(scales(tIdx), rotAngleXYZs(tIdx,:),...
                        transVecXYZs(:,tIdx), tIdx);
                end
                if verbose == true
                    fprintf("pixelError:%.4f, teethPoseError%.4f\n", pixelError, teethPoseError);
                end
                losses(ph) = pixelError + aniScaleError + teethPoseError;
            end
            loss = sum(obj.weightPhotos.*losses);
        end
        


        % 将需要优化的参数合并为一行向量
        function [x0, pIdx] = getCurrentParamsOf5Views_as_x0(obj, stage, step)
            % argument 'step' works only in stage2
            pIdx = containers.Map(["ex_rxyz","ex_txyz","focLth","dpix","u0","v0",...
                "rela_rxyz","rela_txyz"], [1,16,31,36,41,46,51,54]); % 与x0相对应
            x0 = horzcat(reshape(obj.ex_rxyz, [1,15]), reshape(obj.ex_txyz, [1,15]),...
                obj.focLth, obj.dpix, obj.u0, obj.v0, obj.rela_rxyz, obj.rela_txyz');
            if stage == 1 % 优化全局参数和牙列尺寸
                pIdx("rowScaleXZ") = length(x0) + 1;
                x0 = horzcat(x0, obj.rowScaleXZ);
            elseif stage == 2 % 优化全局参数和牙齿局部位姿及尺寸
                switch step
                    case 1
                        pIdx("tXYZs") = length(x0) + 1;
                        x0 = horzcat(x0, reshape(obj.transVecXYZs,1,[]));
                    case 2
                        pIdx("rXYZs") = length(x0) + 1;
                        x0 = horzcat(x0, reshape(obj.rotAngleXYZs,1,[]));
                    case 3
                        pIdx("scales") = length(x0) + 1;
                        x0 = horzcat(x0, obj.scales);
                    case 4
                        pIdx("tXYZs") = length(x0) + 1;
                        pIdx("rXYZs") = length(x0) + 3*obj.NumAllTeeth + 1;
                        pIdx("scales") = length(x0) + 6*obj.NumAllTeeth + 1;
                        x0 = horzcat(x0, reshape(obj.transVecXYZs,1,[]), reshape(obj.rotAngleXYZs,1,[]), obj.scales);
                end
            end
        end

        function [lb, ub] = getBoundsOnParams(obj, stage, step)
            ex_rxyz_d = 0.5;
            ex_txyz_lb = repmat([-20,-20,30], [1,5]);
            ex_txyz_ub = repmat([20,20,100], [1,5]);
            focLth_lb = 15 * ones([1,5]);
            focLth_ub = 75 * ones([1,5]);
            u0 = 400 * ones([1,5]);
            v0 = 300 * ones([1,5]);
            u0_d = 100;
            v0_d = 100;
            dpix_d = 0.03;
            rela_rxyz_lb = -0.1 * ones([1,3]);
            rela_rxyz_ub = 0.1 * ones([1,3]);
            rela_txyz_lb = [-3, -10, -5];
            rela_txyz_ub = [3, -2, 8];
            lb = horzcat(reshape(obj.ex_rxyz, [1,15])-ex_rxyz_d, ex_txyz_lb,...
                focLth_lb, obj.dpix-dpix_d, u0-u0_d, v0-v0_d, rela_rxyz_lb, rela_txyz_lb);
            ub = horzcat(reshape(obj.ex_rxyz, [1,15])+ex_rxyz_d, ex_txyz_ub,...
                focLth_ub, obj.dpix+dpix_d, u0+u0_d, v0+v0_d, rela_rxyz_ub, rela_txyz_ub);
            if stage == 1 % 优化全局参数和牙列尺寸
                rowScaleXZ_lb = [0.5, 0.5];
                rowScaleXZ_ub = [1.5, 1.5];
                lb = horzcat(lb, rowScaleXZ_lb);
                ub = horzcat(ub, rowScaleXZ_ub);
            elseif stage == 2 % 优化全局参数和牙齿局部位姿及尺寸 % 不设置过强的约束
                switch step
                    case 1
                        bd_lth = Inf * ones(1,3*obj.NumAllTeeth);
                        lb = horzcat(lb, -bd_lth);
                        ub = horzcat(ub, bd_lth);
                    case 2
                        bd_lth = Inf * ones(1,3*obj.NumAllTeeth);
                        lb = horzcat(lb, -bd_lth);
                        ub = horzcat(ub, bd_lth);
                    case 3
                        bd_lth = ones(1,obj.NumAllTeeth);
                        lb = horzcat(lb, 1-bd_lth);
                        ub = horzcat(ub, 1+bd_lth);
                    case 4
                        bd_lth1 = Inf * ones(1, 6*obj.NumAllTeeth);
                        bd_lth2 = ones(1, obj.NumAllTeeth);
                        lb = horzcat(lb, -bd_lth1, 1-bd_lth2);
                        ub = horzcat(ub, bd_lth1, 1+bd_lth2);
                end
            end
        end


        % 根据优化得到的结果进行参数的更新
        function updateParamsOf5Views(obj, params, pIdx, stage, step)
            [obj.ex_rxyz, obj.ex_txyz, obj.focLth, obj.dpix, obj.u0, obj.v0, obj.rela_rxyz, obj.rela_txyz]...
                = EmOptMStep.parseGlobalParamsOf5Views(params, pIdx);
            obj.rowScaleXZ = [1,1];
            if stage == 1
                obj.rowScaleXZ = params(pIdx("rowScaleXZ"):pIdx("rowScaleXZ")+1);
            elseif stage == 2
                switch step
                    case 1
                        obj.transVecXYZs = reshape(params(pIdx("tXYZs"):pIdx("tXYZs")+3*obj.NumAllTeeth-1), [3,obj.NumAllTeeth]);
                    case 2
                        obj.rotAngleXYZs = reshape(params(pIdx("rXYZs"):pIdx("rXYZs")+3*obj.NumAllTeeth-1), [obj.NumAllTeeth,3]);
                    case 3
                        obj.scales = params(pIdx("scales"):pIdx("scales")+obj.NumAllTeeth-1);
                    case 4
                        [obj.scales, obj.rotAngleXYZs, obj.transVecXYZs] =...
                            EmOptMStep.parseTeethPoseParams(params, pIdx, obj.NumAllTeeth);
                end
            end
        end
        
        % 根据当前优化参数重建牙齿三维点云,暂不考虑牙齿形状空间（不考虑上下牙列位置关系）
        function X_pred = reconstruct3D(obj)
            [~,~,numTeeth] = size(obj.X_Mu);
            X_pred = zeros(size(obj.X_Mu));
            rotMats = EmOptMStep.computeRotMats(obj.rotAngleXYZs);
            for i = 1:numTeeth
                X_pred(:,:,i) = obj.scales(i) * rotMats(:,:,i) * (obj.X_Mu(:,:,i) - obj.X_Mu_centroids(:,i))...
                    + obj.transVecXYZs(:,i) + obj.X_Mu_centroids(:,i);
            end
            X_pred = [obj.rowScaleXZ(1);1;obj.rowScaleXZ(2)] .* X_pred;
        end

    end

    
    % 静态方法
    methods(Static)
        % 左乘旋转矩阵 convert euler angle rXYZ to left-multiplication matrix
        function rotMats = computeRotMats(rxyz)
            rotMats = eul2rotm(rxyz(:,end:-1:1),"ZYX"); % size = (3,3) or (3,3,?)
        end
        
        % 相机投影矩阵
        function intrProjMat = IntrinsicProjectionMatrix(fL, dp, u0, v0)
            intrProjMat = [[fL/dp, 0., u0]; [0., fL/dp, v0]; [0., 0., 1.]];
        end
        
        function [ex_rxyz, ex_txyz, focLth, dpix, u0, v0, rela_rxyz, rela_txyz] = parseGlobalParamsOf5Views(params, pIdx)
            ex_rxyz = reshape(params(pIdx("ex_rxyz"):pIdx("ex_rxyz")+14), [5,3]); % Matlab reshape is column-first; Fortran-Style 
            ex_txyz = reshape(params(pIdx("ex_txyz"):pIdx("ex_txyz")+14), [3,5]); % transpose
            focLth = params(pIdx("focLth"):pIdx("focLth")+4);
            dpix = params(pIdx("dpix"):pIdx("dpix")+4);
            u0 = params(pIdx("u0"):pIdx("u0")+4);
            v0 = params(pIdx("v0"):pIdx("v0")+4);
            rela_rxyz = params(pIdx("rela_rxyz"):pIdx("rela_rxyz")+2);
            rela_txyz = params(pIdx("rela_txyz"):pIdx("rela_txyz")+2)'; % transpose
        end
        
        function [scales, rotAngleXYZs, transVecXYZs] = parseTeethPoseParams(params, pIdx, numTeeth)
            scales = params(pIdx("scales"):pIdx("scales")+numTeeth-1);
            rotAngleXYZs = reshape(params(pIdx("rXYZs"):pIdx("rXYZs")+3*numTeeth-1), [numTeeth,3]);
            transVecXYZs = reshape(params(pIdx("tXYZs"):pIdx("tXYZs")+3*numTeeth-1), [3,numTeeth]);
        end
        
        % M-step stage1 non-linear constraint
        function [c,ceq] = rowScaleNonLConstr(x)
            rowScaleX = x(end-1);
            rowScaleZ = x(end);
            eps = 0.01;
            c = [min(rowScaleX,rowScaleZ)-eps-1; 1-max(rowScaleX,rowScaleZ)-eps];
%             c = 2 * min(abs(1-rowScaleX),abs(1-rowScaleZ)) - abs(rowScaleX-rowScaleZ);
            ceq = [];
        end
    end
end