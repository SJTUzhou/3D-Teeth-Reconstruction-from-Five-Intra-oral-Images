function parallel_run_GP_Non_Rigid_Registration(mat2Load, mat2Save)
    % parallel run directly in matlab
    % load params into .mat file
    load(mat2Load, "np_srcX", "np_targetYs", "s", "sigma", "n", "eta");

    gpReg = GP_Non_Rigid_Registration(np_srcX, s, sigma, n);
    gpReg.compute_EigVals_EigFuncs_of_GP_K(); % 对高斯形变场进行低秩近似

    n_sample = length(np_targetYs);
    targetYs = cell(1,n_sample); % 待非刚性的配准的点云
    for i = 1:n_sample
        targetYs{i} = double(np_targetYs{i})';
    end
    eta = double(eta);% normalization factor of alpha


    deformedXs = zeros([gpReg.d, gpReg.N, n_sample]);
    
    
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [];
    ub = [];
    nonlcon = [];
    options = optimoptions('fmincon','MaxFunctionEvaluations',100);
    srcX = gpReg.srcX;
    GP_Mu = gpReg.GP_Mu;
    phi_n = gpReg.phi_n;
    lambda_n = gpReg.lambda_n;
    alpha0 = zeros([n,1]);

    alpha_opt = zeros([n,n_sample]);
    num_thread = 4;
    % parallel opt
    parfor (idx = 1:n_sample, num_thread)
        alpha_opt(:,idx) = fmincon(@(alpha)GP_Non_Rigid_Registration.registration_loss(alpha,...
            srcX, targetYs{idx}, GP_Mu, phi_n, lambda_n, eta), alpha0, A,b,Aeq,beq,lb,ub,nonlcon,options);

        deformedXs(:,:,idx) = GP_Non_Rigid_Registration.getDeformedX(alpha_opt(:,idx), srcX, GP_Mu, phi_n, lambda_n);

        fprintf('The optimization of No.%d completed\n',idx);
    end

    % save params into .mat file
    save(mat2Save, "deformedXs");
end