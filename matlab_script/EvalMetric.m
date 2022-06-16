classdef EvalMetric
    % 评估当前优化结果的指标

    properties
        % 空
    end

    methods
        % 构造函数（空）
        function obj = EvalMetric()
        end
    end

    % 静态方法
    methods(Static)
        function RMSE = computeRMSE(X_ref, X_pred)
            % 计算 Root Mean Squared Surface Distance
            % 计算X_ref, X_pred对应点之间的平均欧氏距离，X_ref, X_pred点数量相等且存在一一对应的关系
            % size(X_ref) = size(X_pred) = (3,1500,?)
            X_ref_stacked = reshape(X_ref, 3, []);
            X_pred_stacked = reshape(X_pred, 3, []);
            RMSE = mean(vecnorm(X_ref_stacked - X_pred_stacked));
        end

        function ASSD = computeASSD(X_ref, X_pred)
            % 计算 Average Symmetric Surface Distance 平均表面距离
            % https://blog.csdn.net/qq_33854260/article/details/115000187
            [~,numPoint,numTeeth] = size(X_ref);
            ASSDs = zeros(1,numTeeth);
            for i = 1:numTeeth
                D = zeros(numPoint, numPoint);
                for j = 1:numPoint
                    D(j,:) = vecnorm(X_pred(:,:,i)-X_ref(:,j,i)); % 对于牙齿i,X_ref中一点到X_pred中所有点的距离
                end
                d1 = min(D,[],1);
                d2 = min(D,[],2);
                ASSDs(i) = (mean(d1) + mean(d2)) / 2;
            end
            ASSD = mean(ASSDs);
        end

        function HD = computeHD(X_ref, X_pred)
            % 计算 Hausdorff Distance(豪斯多夫距离)
            % https://en.wikipedia.org/wiki/Hausdorff_distance
            [~,numPoint,numTeeth] = size(X_ref);
            HDs = zeros(1,numTeeth);
            for i = 1:numTeeth
                D = zeros(numPoint, numPoint);
                for j = 1:numPoint
                    D(j,:) = vecnorm(X_pred(:,:,i)-X_ref(:,j,i));
                end
                d1 = min(D,[],1);
                d2 = min(D,[],2);
                HDs(i) = max(max(d1), max(d2));
            end
            HD = max(HDs);
        end
    end
end