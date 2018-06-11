clc;clear;close all;
format compact;
%% Parameter setting
d0 = 1;%单位m
Pd0 = 31.7;%单位db，测量后的值
PT = 0;%单位dbm,发射功率
n = 2.2;%路径损耗指数
sigma= 3;%增加的0均值高斯过程的标准差
N = 1000;%收集了100组数据
%% The setting of Reader, Reference tag and Test tag
% 在空间的角落放置4个阅读器
PosReader = [-0.5,-0.5;-0.5,10.5;10.5,-0.5;10.5,10.5];
nPosTag = 121;
PosTag = zeros(nPosTag, 2);
% 在 11 x 11 的空间中放置 121 个参考标签
%每个标签相距 1 m，从坐标(0,0)到坐标(10,10)  
for iRow = 1:11
   for jCol = 1:11
      PosTag((iRow-1)*11+jCol,:) = [jCol-1 iRow-1]; 
   end
end
nPosValTag = 121;
PosValTag = 10 * rand(nPosValTag, 2);
% 在空间中随机生成1000个测试标签，用于测试定位效果。
nPosTestTag = 1000;
PosTestTag = 10 * rand(nPosTestTag, 2);    % 随机产生测试标签的位置
NumberOfReader = size(PosReader, 1);    % 阅读器的数量
NumberOfTag = size(PosTag, 1);          % 参考标签的数量
NumberOfValTag = size(PosValTag, 1);    % 验证标签的数量
NumberOfTestTag = size(PosTestTag, 1);  % 测试标签的数量
PR = zeros(NumberOfTag, NumberOfReader, N);        % 参考标签RSSI值
PR_Val = zeros(NumberOfValTag, NumberOfReader, N); % 验证标签RSSI值 
PR_Test = zeros(NumberOfTestTag, NumberOfReader, N); % 测试标签RSSI值

%% Calculate the distances
  % 参考标签与阅读器之间的距离
[d_RT] = calDistance(NumberOfTag, NumberOfReader, PosTag, PosReader);
% 计算验证标签到阅读器的距离
[d_RTV]=calDistance(NumberOfValTag, NumberOfReader, PosValTag, PosReader);
 % 测试标签与阅读器之间的距离
[d_RTT]=calDistance(NumberOfTestTag, NumberOfReader, PosTestTag, PosReader);

%% Experiment setting

Iters = 50;  % 最大迭代次数
Error = zeros(Iters,1);  % 每次迭代的输出误差
averageError = zeros(80, 1);

OutputOfTrain = zeros(NumberOfTag,2,Iters);
OutputOfTest = zeros(NumberOfTestTag,2,Iters);
%% Start running
nCount = 0;
for hidden = 1
    nCount = nCount + 1;
for iter = 1:Iters
%============================计算RSSI值============================%
    Para = [PT, Pd0, d0, n, sigma, N]; % RSSI参数设置
    [PR PR_Val PR_Test] = calPR(NumberOfTag, NumberOfValTag, ...
             NumberOfTestTag, NumberOfReader, d_RT, d_RTV, d_RTT,Para);
%==========================数据预处理===============================%
%================高斯滤波===================%
    [PRFilter]=GaussianFilter(PR,NumberOfTag,NumberOfReader,N);
    [PRValFilter]=GaussianFilter(PR_Val,NumberOfValTag,NumberOfReader,N);
    [PRTestFilter]=GaussianFilter(PR_Test,NumberOfTestTag,NumberOfReader,N);
    % 归一化处理
    PRGY = [PRFilter;PRValFilter;PRTestFilter];
    PRGYMAX = max(PRGY, [], 1);
    PRGYMIN = min(PRGY, [], 1);
    PRGY = (PRGY - PRGYMIN(ones(size(PRGY, 1), 1), :))./...
        (PRGYMAX(ones(size(PRGY, 1), 1), :) - PRGYMIN(ones(size(PRGY, 1), 1), :));
    [GYrow,GYcol] = size(PRGY);
    % 划分预处理过后的训练集、验证集和测试集
    TrainInput = PRGY(1:NumberOfTag,:);
    ValInput = PRGY(NumberOfTag+1:NumberOfTag+NumberOfValTag,:);
    TestInput = PRGY(NumberOfTag + NumberOfValTag+1:end,:);
    %% ELM
    % Initialize the parameters of ELM
    NumberOfHidden = 31;    % 隐层节点个数
%     epsilon_init = sqrt(6)./sqrt(4+NumberOfHidden);
    epsilon_init = 1;
    InputWeight = 2*rand(NumberOfHidden, 4)*epsilon_init-epsilon_init;% 初始化输入权重
    HiddenBias = 2*rand(NumberOfHidden,1)*epsilon_init-epsilon_init; % 初始化隐层神经元偏置
    % 直接使用ELM
    [OutputOfTrain(:,:,iter),OutputOfTest(:,:,iter),trainTime,testTime]= ...
        ELM(PosTag,TrainInput, TestInput, NumberOfHidden, InputWeight, HiddenBias);
    
    % 计算定位误差
    temp = 0;
    for i = 1:NumberOfTestTag
        temp = temp + norm(OutputOfTest(i,:,iter) - PosTestTag(i, :));
    end
    Error(iter,1) = temp ./ NumberOfTestTag;
end
    x(nCount, 1) = hidden;    % 损失路径常数
    averageError(nCount,:) = mean(Error,1);   % 平均误差
    mean(Error,1)
end
[a,i] = min(averageError);
i + 20 
%% plot figure
figure();
plot(x, averageError, 'bo-', 'MarkerSize', 6, 'LineWidth', 1);
xlabel('hidden number');
ylabel('MeanError (m)');
title(sprintf('n=%d,sigma=%d', N,sigma));
