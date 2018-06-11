function[xm,fv] = PSO(fitness, sampleN, c1, c2, w, M, D, sigma, ...
    InputWeight_init, HiddenBias_init)
%==========================================================================
% 功能：粒子群算法。
% 参数：fitness 适应度函数。
%       sampleN 输入样本数量。
%       c1 学习因子1。更新速度公式认知部分参数。
%       c2 学习因子2。更新速度公式社会部分参数
%       w 惯性权重。更新速度公式惯量部分参数。
%       M 迭代次数。
%       D 每个粒子群大小。
% 返回值：xm 全局最优解。
%        fv 最优解对应适应度值。
% 备注：
% 日期：20180418
%==========================================================================
%%%%%%%%%%%给定初始化条件%%%%%%%%%%%%
%c1 学习因子1
%c2 学习因子2
%w 惯性权重
%M 最大迭代次数
%D 搜索空间维数
%N 初始化群体个体数目
%%%%%%%%初始化种群的个体（可以在这里限定位置和速度的范围） %%%%%%%%%%%
format long;
for iraw = 1:sampleN
    for jcol = 1:D
        x(iraw,jcol) = 2 * sigma * rand - sigma;      %随机初始化位置
        v(iraw,jcol) = 2 * sigma * rand - sigma;      %随机初始化速度
    end
end
% x(1, :) = [InputWeight_init(:); HiddenBias_init(:)]';
for iraw = 1:sampleN
    p(iraw) = fitness(x(iraw,:));
    y(iraw,:) = x(iraw,:);
end
pg = x(sampleN,:);                        %Pg为全局最优
for iraw = 1:(sampleN-1)
    if fitness(x(iraw,:)) < fitness(pg)
        pg = x(iraw,:);
    end
end
%%%%%%%%%%%%%进入主要循环，按照公式依次迭代，直到满足精度要求%%%%%%%%%
for t = 1:M
    for i = 1:sampleN                     %更新速度、位移
        v(i,:) = w * v(i,:) + c1 * rand *(y(i,:) - x(i,:)) + c2 * rand *...
          (pg - x(i,:));
        x(i,:) = x(i,:) + v(i,:);
        if fitness(x(i,:)) < p(i)
            p(i) = fitness(x(i,:));     % 第i个粒子个体极值
            y(i,:) = x(i,:);            % 更新最优个体
        end
        if p(i) < fitness(pg)
            pg = y(i,:);        % 更新全局最优值
        end
    end
    Pbest(t) = fitness(pg);
end
xm = pg';            % 全局最优解
fv = fitness(pg);    % 最优解对应适应度值