function [error] = RBFILS(PosTag,TrainInput, TestInput,NumberOfHidden, PosTestTag)
global dimension;
AmtTag = size(TrainInput, 1);
AmtTestTag = size(TestInput, 1);
%%%%%%%%%%%%%训练网络%%%%%%%%%%%%%%%%%%%%%%%%%
N_cluster = NumberOfHidden;        
%%%%%%%%%%%%%计算中心向量%%%%%%%%%%%%%%%%%%%%%
[center,U,obj_fcn] = fcm(TrainInput,N_cluster); %模糊聚类确定中心值
%确定隐元宽度，即RBF方差，求解各个中心的平均宽度。
% for i = 1:N_cluster
%     eucenter = zeros(N_cluster, 1);
%     for j = 1:N_cluster
%         eucenter(j, 1) = norm(center(i,:) - center(j,:));
%     end
%     % 每个RBF核的宽度
%     rbfvar(i, 1) = sum(eucenter);
% end      
for i = 1:N_cluster
    eucenter=0;
    for j = 1:N_cluster
        if i~=j
            eucenter = eucenter+norm(center(i,:)-center(j,:));           
        end
    end
    rbfvar(i,1)=mean(eucenter);
end

%%%%%%%%%%%%%计算输出权重%%%%%%%%%%%%%%%%%%%%%
for i = 1:N_cluster
    for k = 1:AmtTag
        G(k,i) = exp((-1/(2*rbfvar(i,1)^2))*norm(TrainInput(k,:)-center(i,:)).^2);
    end
end
rbfweight = pinv(G)*(PosTag);
%%%%%%%%%%%%%计算训练误差%%%%%%%%%%%%%%%%%%%%%
for k = 1:AmtTag
    tempy1=0;
    tempy2=0;
    for i = 1:N_cluster
        tempy1 = rbfweight(i,1).*exp((-1/(2*rbfvar(i,1)^2))*...
            norm(TrainInput(k,:)-center(i,:)).^2)+tempy1;
        tempy2 = rbfweight(i,2).*exp((-1/(2*rbfvar(i,1)^2))*...
            norm(TrainInput(k,:)-center(i,:)).^2)+tempy2;
    end
    y(k,1)= tempy1; %训练输出
    y(k,2)= tempy2;
end
eutemp=0;

%%%%%%%%%%%%%计算测试误差%%%%%%%%%%%%%%%%%%%%
for k = 1:AmtTestTag
    tempy1=0;
    tempy2=0;
    for i = 1:N_cluster
        tempy1 = rbfweight(i,1).*exp((-1/(2*rbfvar(i,1).^2))*...
            norm(TestInput(k,:)-center(i,:)).^2)+tempy1;
        tempy2 = rbfweight(i,2).*exp((-1/(2*rbfvar(i,1).^2))*...
            norm(TestInput(k,:)-center(i,:)).^2)+tempy2;
    end
    resrbf1(k,1)= tempy1;
    resrbf1(k,2)= tempy2;
end

% 计算损失值
error(1,1) = calLoss(AmtTag, y, PosTag); % 训练误差
error(1,2) = calLoss(AmtTestTag, resrbf1, PosTestTag); % 测试误差

% [best_ab,best_fval]=...
%     optainet(center,N_cluster,AmtTag,TrainInput,PosTag,AmtTag,TrainInput,...
%     PosTag,AmtTestTag,TestInput,PosTestTag);
%  %由最优抗体确定RBF中心
%         center=reshape(best_ab,dimension ,N_cluster)';
%          %确定隐元宽度，即RBF方差，求解各个中心的平均宽度。
%          for i = 1:N_cluster
%             eucenter = zeros(N_cluster, 1);
%             for j = 1:N_cluster
%                 eucenter(j, 1) = norm(center(i,:) - center(j,:));
%             end
%             rbfvar(i, 2) = sum(eucenter);
%          end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%通过伪逆确定权值%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         for i = 1:(N_cluster)
%             for k = 1:AmtTag
%                 G(k,i) = exp((-1/(2*rbfvar(i,2)^2))*norm(TrainInput(k,:)-center(i,:)).^2);
%             end
%         end
%         rbfweight = pinv(G)*(PosTag);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%训练完毕使用神经网络%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         %-----------------%
%         %第一层RBF输出构造-----------------------%
%         for k = 1:AmtTestTag
%             tempy1=0;
%             tempy2=0;
%             for i = 1:N_cluster
%                 tempy1 = rbfweight(i,1).*exp((-1/(2*rbfvar(i,2)^2))*...
%                     norm(TestInput(k,:)-center(i,:)).^2)+tempy1;
%                 tempy2 = rbfweight(i,2).*exp((-1/(2*rbfvar(i,2)^2))*...
%                     norm(TestInput(k,:)-center(i,:)).^2)+tempy2;
%             end
%             resrbf2(k,1)= tempy1;   %测试集实际输出结果
%             resrbf2(k,2)= tempy2;
%         end
%         error(1,2) = calLoss(AmtTestTag, resrbf2, PosTestTag);