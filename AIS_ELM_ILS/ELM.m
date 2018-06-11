function [OutputOfTrain,OutputOfTest,trainEndTime,testEndTime] = ...
    ELM(PosTag,TrainInput, TestInput, NumberofHidden,InputWeight,HiddenBias)
    C = 10;    % 正则化系数
    NumberofTag = size(TrainInput, 1);
    NumberofTestTag = size(TestInput, 1);
    
    tic;       % 训练计时开始
%% training phase
    tempH = TrainInput * InputWeight';
    ind = ones(1, NumberofTag);
    tempH = tempH + HiddenBias(:,ind)';
    H = 1 ./ (1 + exp(-tempH)); % sigmoid 激活函数
    
    %----------------------------计算输出权重----------------------------%
    OutputWeight=pinv(H) * PosTag;
%     OutputWeight = pinv(H'*H + 1./C) * H' * PosTag;
    OutputOfTrain = H * OutputWeight;
    trainEndTime = toc; % 训练计时结束
%% testing phase
    tic;              % 测试开始计时

    tempH1 = TestInput * InputWeight';
    tempH1 = tempH1 + HiddenBias(:, ones(1, NumberofTestTag))';
    H1 = 1./(1+exp(-tempH1));
    OutputOfTest = H1 * OutputWeight;     % 测试集实际输出
    testEndTime = toc; % 测试结束计时
    
end