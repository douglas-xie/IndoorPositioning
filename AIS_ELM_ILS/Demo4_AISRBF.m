clc;clear;close all;
format compact;
addpath('RBF');addpath('AISRBF');
%% Parameter setting
d0 = 1;%单位m
Pd0 = 31.7;%单位db，测量后的值
PT = 0;%单位dbm,发射功率0
n = 2.2;%路径损耗指数
sigma= 3;%增加的0均值高斯过程的标准差
N = 1000;%收集了100组数据
%% The setting of Reader, Reference tag and Test tag
% 在空间的角落放置4个阅读器
NumberOfReader =  4;
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
% 随机生成121个验证标签
% NumberofValTag = 121;    % 验证标签的数量
% PosValTag = 10 * rand(NumberofValTag, 2);
NumberofValTag = 121;
PosValTag = PosTag;
% 在空间中随机生成1000个测试标签，用于测试定位效果。
NumberofTestTag = 1000;     % 测试标签的数量
PosTestTag = 10 * rand(NumberofTestTag, 2);    % 随机产生测试标签的位置

PR = zeros(NumberofTag, NumberOfReader, N);        % 参考标签RSSI值
PR_Val = zeros(NumberofValTag, NumberOfReader, N); % 验证标签RSSI值 
PR_Test = zeros(NumberofTestTag, NumberOfReader, N); % 测试标签RSSI值

%% 查看标签和阅读器的位置
% plot(PosTag(:,1), PosTag(:,2), 'go', 'MarkerSize', 8);
% hold on;
% plot(PosReader(:, 1), PosReader(:, 2), 'ks', 'MarkerFaceColor', 'r', 'MarkerSize', 12);
% plot(PosTestTag(:, 1), PosTestTag(:, 2), 'ko', 'MarkerSize', 6, 'MarkerFaceColor', 'b');
% % set(gca,'XTickLabel','','YTickLabel','');
% % set(gca,'XTick','','YTick','');
% axis([-0.6 10.6 -0.6 10.6]);

%% Calculate the distances
  % 参考标签与阅读器之间的距离
[d_RT] = calDistance(NumberofTag, NumberOfReader, PosTag, PosReader);
% 计算验证标签到阅读器的距离
[d_RTV]=calDistance(NumberofValTag, NumberOfReader, PosValTag, PosReader);
 % 测试标签与阅读器之间的距离
[d_RTT]=calDistance(NumberofTestTag, NumberOfReader, PosTestTag, PosReader);

%% Experiment setting

Iters = 10;  % 最大迭代次数
Error = zeros(Iters,2);  % 每次迭代的输出误差
averageError = zeros(10, 2);

OutputofTrain = zeros(NumberofTag,2,Iters);
OutputofTest = zeros(NumberofTestTag,2,Iters);
%% Start running
nCount = 0;  % 损失路径的个数
for n = 1:10
    n
    nCount = nCount + 1;
for iter = 1:Iters
%============================计算RSSI值============================%
    Para = [PT, Pd0, d0, n, sigma, N]; % RSSI参数设置
    [PR, PR_Val, PR_Test] = calPR(NumberofTag, NumberofValTag, ...
             NumberofTestTag, NumberOfReader, d_RT, d_RTV, d_RTT,Para);
%==========================数据预处理===============================%
%================高斯滤波===================%
    [PRFilter]=GaussianFilter(PR,NumberofTag,NumberOfReader,N);
    [PRValFilter]=GaussianFilter(PR_Val,NumberofValTag,NumberOfReader,N);
    [PRTestFilter]=GaussianFilter(PR_Test,NumberofTestTag,NumberOfReader,N);
    [PRFilter, PRValFilter, PRTestFilter] = addDeltaPR(PRFilter, PRValFilter, PRTestFilter);
    % 归一化处理
    PRGY = [PRFilter;PRValFilter;PRTestFilter];
    [PRGY, PRGYMIN, PRGYMAX] = normalPR(PRGY);
    [GYrow,GYcol] = size(PRGY);
    % 划分训练集、验证集和测试集
    TrainInput = PRGY(1:NumberofTag,:);
%     ValInput = PRGY(NumberofTag+1:NumberofTag+NumberofValTag,:);
    ValInput = TrainInput;
    TestInput = PRGY(NumberofTag + NumberofValTag+1:end,:);
    %% 保存数据集
%     save('DataBase\Val', 'ValInput', 'NumberofValTag', 'PosValTag');
%     save('DataBase\Train', 'TrainInput', 'NumberofTag', 'PosTag');
%     save('DataBase\Test', 'TestInput', 'NumberofTestTag', 'PosTestTag');
    
    %% ELM
    % Initialize the parameters of ELM
    NumberofHidden = 21;    % 隐层节点个数
    epsilon_init = sqrt(6)./sqrt(4+NumberofHidden);
%     epsilon_init = 1;
    InputWeight_init = 2*rand(NumberofHidden, 7)*epsilon_init-epsilon_init;% 初始化输入权重
    HiddenBias_init = 2*rand(NumberofHidden,1)*epsilon_init-epsilon_init; % 初始化隐层神经元偏置
    
    %% 使用RBF
    [error]=RBFILS(PosTag,TrainInput,TestInput,NumberofHidden,PosTestTag);
    Error(iter, 1) = error(1,1);
%     Error(iter, 2) = error(1,2);
end
    x(nCount, 1) = n;    % 损失路径常数
    averageError(nCount,:) = mean(Error,1)   % 平均误差
end

%% plot figure
figure;
plot(x,averageError(:,1),'go--','MarkerSize',8);hold on;
plot(x,averageError(:,2),'bd--','MarkerSize',8);
% plot(x,averageError(:,3),'rx--','MarkerSize',8);
% plot(x,averageError(:,4),'ks--','MarkerSize',8);
xlabel('The signal loss constant(\alpha)');
ylabel('Average positioning error');
% legend('ELM(sp = 2.0m)', 'AIS-ELM(sp = 2.0m)', 'PSO-ELM(sp = 2.0m)', ...
%     'RBF(sp = 2.0m)');
legend('RBF(sp = 1.0m)', 'AIS-RBF(sp = 1.0m)');
title(sprintf('N=%d,sigma=%f', N, sigma));
