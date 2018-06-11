function aff = affinity(AbPop, NumberofHidden, NumberofTag, TrainInput,...
    PosTag, NumberofValidation, ValidationInput, PosValidation)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 计算AIS算法种群每个个体的亲和力，AbPop 为需要计算的种群，返回aff 种群亲和力，
% aff 值越大，所代表的个体更优。
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[row, col] = size(AbPop);
NumberofInput = size(TrainInput, 2);
C = 0.1;
aff = zeros(row, 1);
for irow = 1:row
   InputWeight = reshape(AbPop(irow,1:NumberofHidden*NumberofInput)', ...
     NumberofHidden, NumberofInput);
   HiddenBias = reshape(AbPop(irow,NumberofHidden*NumberofInput+1:end)',...
       NumberofHidden, 1);  
   %% training phase
    tempH = TrainInput * InputWeight';
    ind = ones(1, NumberofTag);
    tempH = tempH + HiddenBias(:,ind)';
    H = 1 ./ (1 + exp(-tempH)); % sigmoid 激活函数
    H = logsig(tempH);
    %----------------------------计算输出权重----------------------------%
    OutputWeight=pinv(H) * PosTag;
%     OutputWeight = pinv(H'*H + 1./C) * H' * PosTag;
%     OutputOfTrain = H * OutputWeight;
%% 计算定位误差
    tempH1 = ValidationInput * InputWeight';
    tempH1 = tempH1 + HiddenBias(:, ones(1, NumberofValidation))';
    H1 = 1./(1+exp(-tempH1));
    Output = H1 * OutputWeight;

    Error = calLoss(NumberofValidation, Output, PosValidation);
    aff(irow, 1) = 1./(1+Error);
end