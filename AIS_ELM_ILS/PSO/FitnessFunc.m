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
for k = 1:m
InputWeight = reshape(x(k,1:inputnum*hiddennum),hiddennum,inputnum);
HiddenBias = reshape(x(k,inputnum*hiddennum+1:inputnum*hiddennum+hiddennum),...
    hiddennum,1);

   % 使用ELM估计坐标值，用以计算误差
   Output=ELM(outputn,inputn, valinput, hiddennum,InputWeight,HiddenBias);

% 计算损失值
    error(k, 1) = calLoss(size(valinput, 1),Output, valoutput);
end
end