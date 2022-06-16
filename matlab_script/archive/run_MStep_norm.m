% run from python script

function run_MStep(stage, maxFuncEval, mat2Load, mat2Save)
    % load params into .mat file
    load(mat2Load, 'np_X_Mu', 'np_X_Mu_pred', 'np_X_Mu_pred_normals',...
        'np_visIdx', 'np_corre_pred_idx', 'np_P_true', 'np_ex_rxyz', 'np_ex_txyz', 'np_focLth', 'np_dpix',...
        'np_u0', 'np_v0', 'np_rela_rxyz', 'np_rela_txyz', 'np_rowScaleXZ', 'np_scales',...
        'np_rotAngleXYZs', 'np_transVecXYZs', 'np_invCovMatOfScale', 'np_invCovMatOfPose');
    
    emoptMstep = EmOptMStep(np_X_Mu, np_X_Mu_pred, np_X_Mu_pred_normals,...
                    np_visIdx, np_corre_pred_idx, np_P_true, np_ex_rxyz, np_ex_txyz, np_focLth, np_dpix,...
                    np_u0, np_v0, np_rela_rxyz, np_rela_txyz, np_rowScaleXZ, np_scales,...
                    np_rotAngleXYZs, np_transVecXYZs, np_invCovMatOfScale, np_invCovMatOfPose);
    
    
    
    % M-step optimization
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [];
    ub = [];
    nonlcon = [];
    options = optimoptions('fmincon','Display',"iter",'Algorithm','interior-point','MaxFunctionEvaluations',maxFuncEval);
    % options = optimoptions('fmincon','Display',"iter",'Algorithm','sqp','MaxFunctionEvaluations',maxFuncEval);
    
    switch stage
        case {0,1} % optimize global parameters with rowScaleXZ
            step = -1;
            [x0, pIdx] = emoptMstep.getCurrentParamsOf5Views_as_x0(stage, step); 
            lb = zeros(size(x0));
            ub = ones(size(x0));
%             [lb, ub] = emoptMstep.getBoundsOnParams(stage, step);     
%             if stage == 1
%                 nonlcon = @(x)EmOptMStep.rowScaleNonLConstr(x);
%             end
            [x_opt, M_loss] = fmincon(@(x)emoptMstep.MStepLoss(x,pIdx,stage,step,false),x0,A,b,Aeq,beq,lb,ub,nonlcon,options);
            emoptMstep.updateParamsOf5Views(x_opt, pIdx, stage, step); % update params from x_opt
        case 2 % optimize global parameters with scales, rotAngleXYZs, transVecXYZs
%             step = 4;
%             [x0, pIdx] = emoptMstep.getCurrentParamsOf5Views_as_x0(stage, step);
%             [lb, ub] = emoptMstep.getBoundsOnParams(stage, step);
%             [x_opt, M_loss] = fmincon(@(x)emoptMstep.MStepLoss(x,pIdx,stage,step,false),x0,A,b,Aeq,beq,lb,ub,nonlcon,options);
%             emoptMstep.updateParamsOf5Views(x_opt, pIdx, stage, step); % update params from x_opt
                
            for step = 1:3
                [x0, pIdx] = emoptMstep.getCurrentParamsOf5Views_as_x0(stage, step);
%                 [lb, ub] = emoptMstep.getBoundsOnParams(stage, step);
                [x_opt, M_loss] = fmincon(@(x)emoptMstep.MStepLoss(x,pIdx,stage,step,false),x0,A,b,Aeq,beq,lb,ub,nonlcon,options);
                emoptMstep.updateParamsOf5Views(x_opt, pIdx, stage, step);
            end
    end



    % parse params from x_opt
    [lbd, ubd] = emoptMstep.getBoundsOnParams(stage, step);
    x_opt = x_opt .* (ubd - lbd) + lbd;
    [ex_rxyz, ex_txyz, focLth, dpix, u0, v0, rela_rxyz, rela_txyz] = emoptMstep.parseGlobalParamsOf5Views(x_opt, pIdx);
    rowScaleXZ = emoptMstep.rowScaleXZ; 
    scales = emoptMstep.scales;
    rotAngleXYZs = emoptMstep.rotAngleXYZs;
    transVecXYZs = emoptMstep.transVecXYZs;
    
    % save params into .mat file
    save(mat2Save, "ex_rxyz", "ex_txyz", "focLth", "dpix", "u0", "v0", "rela_rxyz", "rela_txyz", "rowScaleXZ",...
        "scales", "rotAngleXYZs", "transVecXYZs", "M_loss");
end
