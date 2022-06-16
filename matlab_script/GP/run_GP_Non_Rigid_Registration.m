function run_GP_Non_Rigid_Registration(mat2Load, mat2Save)
    % run from python script
    % load params into .mat file
    load(mat2Load, "np_srcX", "np_targetYs", "s", "sigma", "n", "eta");

    gpReg = GP_Non_Rigid_Registration(np_srcX, np_targetYs, s, sigma, n, eta);
    gpReg.compute_EigVals_EigFuncs_of_GP_K(); % 对高斯形变场进行低秩近似
    
    
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [];
    ub = [];
    nonlcon = [];
    % options = optimoptions('fmincon','Display','iter','MaxFunctionEvaluations',2000);
    options = optimoptions('fmincon','MaxFunctionEvaluations',1000);
    alpha0 = zeros([n,1]);
    for idx = 1:gpReg.n_sample
        alpha_opt = fmincon(@(alpha)gpReg.registration_loss(alpha,idx),...
            alpha0, A,b,Aeq,beq,lb,ub,nonlcon,options);

        gpReg.updateDeformedXs(alpha_opt, idx);
        fprintf('The GP optimization of No.%d completed\n',idx);
    end

   deformedXs = gpReg.deformedXs;
    % save params into .mat file
    save(mat2Save, "deformedXs");
end