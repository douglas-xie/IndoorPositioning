clc;
clear;
% close all;
addpath('AIS');
%% Parameter setting
d0 = 1;%单位m
Pd0 = 31.7;%单位db，测量后的值
PT = 0;%单位dbm,发射功率
n = 2.2;%路径损耗指数 
sigma= 3;%增加的0均值高斯过程的标准差
N = 100;%收集了100组数据
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
% plot(PosTag(:,1), PosTag(:,2), 'ro', 'MarkerSize', 8);
% 在空间中随机生成1000个测试标签，用于测试定位效果。
nPosTestTag = 1000;
PosTestTag = 10 * rand(nPosTestTag, 2);    % 随机产生测试标签的位置
NumberOfReader = size(PosReader, 1);
NumberOfTag = size(PosTag, 1);
NumberOfTestTag = size(PosTestTag, 1);
d_RT = zeros(NumberOfTag, NumberOfReader);
d_RTT = zeros(NumberOfTestTag, NumberOfReader);
PR = zeros(NumberOfTag, NumberOfReader, N); % 4个阅读接收到的参考标签RSSI值
PRtest = zeros(NumberOfTestTag, NumberOfReader, N); % 4个阅读接收到的测试标签RSSI值

%% Calculate the distances
  % 每个参考标签与阅读器之间的距离
for j = 1:NumberOfTag
    for i = 1:NumberOfReader
        d_RT(j,i) = norm(PosTag(j, :) - PosReader(i, :));
    end
end
 % 每个测试标签与阅读器之间的距离
for j = 1:NumberOfTestTag
    for i = 1:NumberOfReader
        d_RTT(j,i) = norm(PosTestTag(j, :) - PosReader(i, :));
    end
end
%% Init the parameter of ELM
NumberOfHidden = 28;    % 隐层节点个数
HiddenBias = sigma .* randn(NumberOfHidden, 1); % 隐层偏置初始化
InputWeight = sigma .* randn(NumberOfReader, NumberOfHidden);% 输入权重初始化
OutputWeight = sigma .* randn(NumberOfHidden, 2);   % 输出权重初始化
%% Experiment setting

Iters = 3;  % 最大迭代次数
Error = zeros(Iters,1);  % 每次迭代的输出误差

OutputOfTrain = zeros(NumberOfTag,2,Iters);
OutputOfTest = zeros(NumberOfTestTag,2,Iters);
%% Start running
for iter = 1:Iters
%============================计算RSSI值============================%
    for j = 1:N
        AddGauss=sigma * randn(NumberOfTag+NumberOfTestTag,NumberOfReader);
        PR(:,:,j)=PT-(Pd0+10.*n.*log10(d_RT./d0)+AddGauss(1:NumberOfTag,:));
        PRtest(:,:,j) = PT - (Pd0 + 10.*n.*log10(d_RTT./d0)+...
            AddGauss(NumberOfTag+1:NumberOfTag+NumberOfTestTag,:));
    end
%==========================数据预处理===============================%
    PR_mean = mean(PR, 3); % 参考标签 RSSI 均值
    PRd_square = zeros(NumberOfTag, NumberOfReader);
    for i = 1:N
        PRd_square = PRd_square + (PR(:,:,i)-PR_mean).^2;
    end
    sigma1 = sqrt(1/(N-1) * PRd_square); % 测试标签 RSSI 值的方差
    
     uplimit = PR_mean + sigma1;    % 滤波上界
     downlimit = PR_mean - sigma1;  % 滤波下界
     PRTemp = zeros(NumberOfTag,NumberOfReader);
     PRFilter = zeros(NumberOfTag,NumberOfReader);
     for i = 1:NumberOfTag
        for j = 1:NumberOfReader
           Length = 0;
           for k = 1:N
              if PR(i,j,k)<uplimit(i,j) && PR(i,j,k)>downlimit(i,j)
                  PRTemp(i,j) = PRTemp(i,j) + PR(i,j,k);
                  Length = Length + 1;
              end
           end
           PRFilter(i,j) = PRTemp(i,j)./Length;    % 参考标签滤波输出
        end
     end
    
 %%%%%%%%%%%%%%%%%测试数据预处理%%%%%%%%%%%%%%%%
 %================高斯滤波===================%
    PRtestmean=mean(PRtest,3);  % 测试标签 RSSI 均值
    PRd_square1 = zeros(NumberOfTestTag, NumberOfReader); % 估计标准差
    for i = 1:N
        PRd_square1 = PRd_square1 + (PRtest(:,:,i)-PRtestmean).^2;
    end
    sigma2 = sqrt(1/(N-1) * PRd_square1);
    %-----筛选出1个标准差范围内的点-----------%
    testuplimit = PRtestmean+sigma2;
    testdownlimit = PRtestmean-sigma2;
    PRtestTemp=zeros(NumberOfTestTag,NumberOfReader);
    PRtestFilter=zeros(NumberOfTestTag,NumberOfReader);
    for i = 1:NumberOfTestTag
      for j=1:NumberOfReader
        Length=0;
        for k=1:N
          if PRtest(i,j,k)<testuplimit(i,j) && PRtest(i,j,k)>testdownlimit(i,j)
             PRtestTemp(i,j)= PRtestTemp(i,j)+PRtest(i,j,k);
             Length=Length+1;
          end
        end
        PRtestFilter(i,j)=PRtestTemp(i,j)./Length;  % 测试标签滤波输出
       end
    end
    
    % 归一化处理
    PRGY = [PRFilter;PRtestFilter];
    PRGYMAX = max(PRGY, [], 1);
    PRGYMIN = min(PRGY, [], 1);
    PRGY = (PRGY - PRGYMIN(ones(size(PRGY, 1), 1), :))./...
        (PRGYMAX(ones(size(PRGY, 1), 1), :) - PRGYMIN(ones(size(PRGY, 1), 1), :));
    [GYrow,GYcol] = size(PRGY);
    % 训练集数据准备
    TrainInput = PRGY(1:GYrow-NumberOfTestTag,:);
    % 验证集数据准备
    NumberOfValidation = NumberOfTag;
    ValidationInput = TrainInput;
    PosValidation = PosTag;
    % 测试集数据准备
    TestInput = PRGY(NumberOfTag+1:NumberOfTestTag+NumberOfTag,:);
    %% Immune system
    % 初始化权重和偏置
    epsilon_init = sqrt(6)./sqrt(4+NumberOfHidden);
    InputWeight = rand(NumberOfHidden, 4)*epsilon_init-epsilon_init;% 初始化输入权重
    HiddenBias = rand(NumberOfHidden,1)*epsilon_init-epsilon_init; % 初始化隐层神经元偏置
    % 使用免疫算法优化
    [best_ab,best_fval,it,best_set,FE]=...
        optainet(InputWeight,HiddenBias,NumberOfHidden,...
        NumberOfTag, PosTag,TrainInput, NumberOfValidation, ...
        PosValidation, ValidationInput);
    
    InputWeight = reshape(best_ab(1:NumberOfHidden*4),NumberOfHidden, 4);
    HiddenBias = reshape(best_ab(NumberOfHidden*4+1:end),NumberOfHidden, 1);
    % 使用 ELM
    [OutputOfTrain(:,:,iter),OutputOfTest(:,:,iter),trainTime,testTime]= ...
        ELM(PosTag,TrainInput, TestInput, NumberOfHidden, InputWeight, HiddenBias);
    time(iter,1) = trainTime;
    time(iter,2) = testTime;
    % 计算测试定位误差
    temp = 0;
    for i = 1:NumberOfTestTag
        temp = temp + norm(OutputOfTest(i,:,iter) - PosTestTag(i, :));
    end
    Error(iter,1) = temp ./ NumberOfTestTag;
end

%% result computing
meanTime = mean(time,1);
minError = max(min(Error))
maxError = max(max(Error))
meanError = mean(mean(Error))
mse = mean(mean(Error.^2))


%% plot figure
% figure();
% for i = 1:NumberOfTestTag
%    h_error = plot([mean(OutputOfTest(i,1,:),3) PosTestTag(i,1)],...
%        [mean(OutputOfTest(i,2,:),3) PosTestTag(i,2)],'b-');hold on; 
% end
% h_rpos = plot(PosTestTag(:,1), PosTestTag(:,2), 'b+', 'LineWidth', 2);hold on;
% h_epos = plot(mean(OutputOfTest(:,1,:),3), mean(OutputOfTest(:,2,:),3),...
%     'ro', 'LineWidth', 2);hold on;
% xlabel('Width of the Area (m)');ylabel('Length of the Area (m)');
% legend([h_rpos h_epos h_error],'Real position of tags',...
%     'Estimated position of tags','Location error');
% axis([0 11 0 7]);
% grid on;
