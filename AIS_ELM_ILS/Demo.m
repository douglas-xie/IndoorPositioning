clc;clear;close all;
format compact;
addpath('utilities');
addpath('AIS');addpath('PSO');addpath('RBF');
%% Parameter setting
sigma= 3;%增加的0均值高斯过程的标准差
N = 100;%收集了100组数据

load(fullfile('DataSet','TrainSet','trainSet_3sigma_100N.mat'));
load(fullfile('DataSet','TestSet','testSet_3sigma_100N.mat'));

NumberofTag = trainSet.NumberofTag;    % 参考标签的数量
PosTag = trainSet.label;

NumberofValTag = NumberofTag;
PosValTag = PosTag;
% 在空间中随机生成1000个测试标签，用于测试定位效果。
NumberofTestTag = testSet.NumberofTag;     % 测试标签的数量
PosTestTag = testSet.label;    % 随机产生测试标签的位置

TrainData = trainSet.Data;
ValData = TrainData;
TestData = testSet.Data;
%% Experiment setting

Iters = 1;  % 最大迭代次数
Error1 = zeros(Iters, 4); % 每次迭代的训练误差
Error2 = zeros(Iters,4);  % 每次迭代的测试误差
trainError = [];
testError = [];

% 训练输出和测试输出结果
OutputofTrain = zeros(NumberofTag,2);
OutputofTest = zeros(NumberofTestTag,2);
%% Start running
for n = 10
    fprintf('The loss path constant is: %d\n', n);
    TrainInput = TrainData{n};
    ValInput = TrainInput;
    TestInput = TestData{n};
for iter = 1:Iters    
    %% ELM
    % Initialize the parameters of ELM
    NumberofHidden = 30;    % 隐层节点个数
    epsilon_init = sqrt(6)./sqrt(4+NumberofHidden); % 设置合理的阈值 
    InputWeight_init = 2*rand(NumberofHidden, NumberofReader)*epsilon_init-epsilon_init;% 初始化输入权重
    HiddenBias_init = 2*rand(NumberofHidden,1)*epsilon_init-epsilon_init; % 初始化隐层神经元偏置
    % 直接使用ELM
    [OutputofTrain,OutputofTest,trainTime,testTime]= ...
        ELM(PosTag,TrainInput,TestInput,NumberofHidden,...
        InputWeight_init,HiddenBias_init);
    % 计算定位误差
    Error1(iter, 1) = calLoss(NumberofTag,OutputofTrain, PosTag); % 训练误差
    [Error2(iter, 1),max_error, min_error] = ...
        calLoss(NumberofTestTag,OutputofTest, PosTestTag);
    %% 使用免疫算法优化
    NumberofHidden = 30;
    InputWeight_init = 2*rand(NumberofHidden, NumberofReader)*epsilon_init-epsilon_init;% 初始化输入权重
    HiddenBias_init = 2*rand(NumberofHidden,1)*epsilon_init-epsilon_init; % 初始化隐层神经元偏置
    [InputWeight_AIS,HiddenBias_AIS]=AIS_ELM(InputWeight_init,HiddenBias_init,...
        NumberofHidden, NumberofTag, PosTag,TrainInput, NumberofValTag, ...
        PosValTag, ValInput);
       % 使用 ELM
    [OutputofTrain,OutputofTest]= ...
        ELM(PosTag,TrainInput,TestInput,NumberofHidden,InputWeight_AIS,HiddenBias_AIS);
    % 计算测试定位误差
    Error1(iter, 2) = calLoss(NumberofTag,OutputofTrain, PosTag); % 训练误差
    Error2(iter, 2) = calLoss(NumberofTestTag,OutputofTest, PosTestTag);
    %% 使用PSO优化
    [InputWeight_PSO, HiddenBias_PSO] = ...
        PSO_ELM(InputWeight_init, HiddenBias_init, NumberofHidden, ...
        PosTag,TrainInput, PosValTag, ValInput);
    % 使用 ELM
    [OutputofTrain,OutputofTest]= ...
        ELM(PosTag,TrainInput,TestInput,NumberofHidden,InputWeight_PSO,HiddenBias_PSO);
    % 计算测试定位误差
    Error1(iter, 3) = calLoss(NumberofTag,OutputofTrain, PosTag); % 训练误差
    Error2(iter, 3) = calLoss(NumberofTestTag,OutputofTest, PosTestTag);
    
    %% 使用RBF
    [error]=RBFILS(PosTag,TrainInput,TestInput,35,PosTestTag);
    Error1(iter, 4) = error(1,1);
    Error2(iter, 4) = error(1,2);
end
    trainError = [trainError; mean(Error1, 1)];  % 平均训练误差
    testError = [testError; mean(Error2, 1)]   % 平均测试误差
end