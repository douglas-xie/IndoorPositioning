function error = FitnessFunc(x,inputnum,hiddennum,inputn,outputn,...
    valinput, valoutput)  
% 该函数用来计算pso适应度值  
%x          input     个体  
%inputnum   input     输入层节点数  
%outputnum  input     隐含层节点数  
%net        input     网络  
%inputn     input     训练输入数据  
%outputn    input     训练输出数据  
  
%error      output    个体适应度值  
  
% 提取 
[m, n] = size(x);
error = zeros(m, 1);
C = 0.1;
for k = 1:m
InputWeight = reshape(x(k,1:inputnum*hiddennum),hiddennum,inputnum);
HiddenBias = reshape(x(k,inputnum*hiddennum+1:inputnum*hiddennum+hiddennum),...
    hiddennum,1);

NumberOfInput = size(inputn, 1);

%% training phase
    tempH = inputn * InputWeight';
    ind = ones(1, NumberOfInput);
    tempH = tempH + HiddenBias(:,ind)';
    H = 1 ./ (1 + exp(-tempH)); % sigmoid 激活函数
    %----------------------------计算输出权重----------------------------%
    OutputWeight=pinv(H) * outputn;
%     OutputWeight = pinv(H'*H + 1./C) * H' * outputn;
    
%% validating phase
    tempH1 = valinput * InputWeight';
    ind = ones(1, size(valinput, 1));
    tempH1 = tempH1 + HiddenBias(:,ind)';
    H1 = 1 ./ (1 + exp(-tempH1)); % sigmoid 激活函数
    
    Output = H1 * OutputWeight;

% 计算损失值
    error(k, 1) = calLoss(size(valinput, 1),Output, valoutput);
end
end