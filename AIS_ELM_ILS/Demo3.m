clc;clear;close all;
format compact;
addpath('AIS');addpath('PSO');addpath('RBF');addpath('DataBase');
%% Parameter setting
d0 = 1;%单位m
Pd0 = 31.7;%单位db，测量后的值
PT = 0;%单位dbm,发射功率0
n = 2.2;%路径损耗指数
sigma= 3;%增加的0均值高斯过程的标准差
N = 100;%收集了100组数据
load('DataBase\Test.mat');
load('DataBase\Train.mat');
load('DataBase\Val.mat');

%% Experiment setting

Iters = 10;  % 最大迭代次数
Error = zeros(NumberofTag,1);  % 每次迭代的输出误差
averageError = zeros(NumberofTag, 1);

%% Start running
nCount = 0;  % 损失路径的个数
ind = randperm(NumberofTag);
TrainInput = TrainInput(ind, :);
PosTag = PosTag(ind, :);
for n = 1:NumberofTag
    trainN = n;
    nCount = nCount + 1;
    train = TrainInput(1:trainN, :);
%============================计算RSSI值============================%

    %% ELM
    % Initialize the parameters of ELM
    NumberofHidden = 28;    % 隐层节点个数
%     epsilon_init = sqrt(6)./sqrt(4+NumberOfHidden);
    epsilon_init = 1;
    InputWeight_init = 2*rand(NumberofHidden, 4)*epsilon_init-epsilon_init;% 初始化输入权重
    HiddenBias_init = 2*rand(NumberofHidden,1)*epsilon_init-epsilon_init; % 初始化隐层神经元偏置
    % 直接使用ELM
    [OutputofTrain,OutputofTest,trainTime,testTime]= ...
        ELM(PosTag(1:trainN, :),train,ValInput,NumberofHidden,...
        InputWeight_init,HiddenBias_init);
    % 计算定位误差
    [Error(n, 1),max_error, min_error] = ...
        calLoss(NumberofValTag,OutputofTest, PosValTag);
    [Error(n, 2),max_error, min_error] = ...
        calLoss(trainN,OutputofTrain, PosTag(1:trainN, :));
    x(nCount, 1) = n;    % 损失路径常数
    averageError = Error;   % 平均误差
end

%% plot figure
figure;
plot(x,averageError(:,1),'go--');hold on;
plot(x,averageError(:,2),'bx--');
xlabel('Training example number');
ylabel('Average positioning error');
legend('CV error', 'Train error');
title('Learning Curve');
