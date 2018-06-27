clc;clear;close all;
format compact;
addpath('utilities');
addpath('AIS');addpath('PSO');addpath('RBF');
%% Parameter setting
d0 = 1;%单位m
Pd0 = 31.7;%单位db，测量后的值
PT = 0;%单位dbm,发射功率0
n = 2.2;%路径损耗指数
sigma= 3;%增加的0均值高斯过程的标准差
N = 100;%收集了100组数据
%% The setting of Reader, Reference tag and Test tag
% 在空间的角落放置4个阅读器
NumberofReader =  4;
PosReader = [-0.5,-0.5;-0.5,10.5;10.5,-0.5;10.5,10.5];
NumberofTag = 121;    % 参考标签的数量
PosTag = zeros(NumberofTag, 2);
% 在 11 x 11 的空间中放置 121 个参考标签
%每个标签相距 1 m，从坐标(0,0)到坐标(10,10)
for iRow = 1:11
   for jCol = 1:11
      PosTag((iRow-1)*11+jCol,:) = [(jCol-1) (iRow-1)]; 
   end
end
% 使用参考标签计算验证误差
NumberofValTag = 121;
PosValTag = PosTag;
% 在空间中随机生成1000个测试标签，用于测试定位效果。
NumberofTestTag = 1000;     % 测试标签的数量
PosTestTag = 10 * rand(NumberofTestTag, 2);    % 随机产生测试标签的位置

PR = zeros(NumberofTag, NumberofReader, N);        % 参考标签RSSI值
PR_Val = zeros(NumberofValTag, NumberofReader, N); % 验证标签RSSI值 
PR_Test = zeros(NumberofTestTag, NumberofReader, N); % 测试标签RSSI值

%% Calculate the distances
  % 参考标签与阅读器之间的距离
[d_RT] = calDistance(NumberofTag, NumberofReader, PosTag, PosReader);
 % 测试标签与阅读器之间的距离
[d_RTT]=calDistance(NumberofTestTag, NumberofReader, PosTestTag, PosReader);

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
for iter = 1:Iters
%============================计算RSSI值============================%
    Para = [PT, Pd0, d0, n, sigma, N]; % RSSI参数设置
    [PR, PR_Test] = calPR(NumberofTag, NumberofTestTag, NumberofReader, ...
        d_RT, d_RTT,Para);
%==========================数据预处理===============================%
%================高斯滤波===================%
    [PRFilter]=GaussianFilter(PR,NumberofTag,NumberofReader,N);
    [PRTestFilter]=GaussianFilter(PR_Test,NumberofTestTag,NumberofReader,N);
    % 归一化处理
    PRGY = [PRFilter; PRTestFilter];
    [PRGY, PRGYMIN, PRGYMAX] = normalPR(PRGY);
    [GYrow,GYcol] = size(PRGY);
    % 划分训练集、验证集和测试集
    TrainInput = PRGY(1:NumberofTag,:);
    ValInput = TrainInput;
    TestInput = PRGY(NumberofTag + 1:end,:);
    
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