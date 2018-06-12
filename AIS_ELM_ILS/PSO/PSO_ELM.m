function [InputWeight, HiddenBias] = ...
    PSO_ELM(InputWeight_init, HiddenBias_init, ...
    hiddennum, PosTag,TrainInput,PosValTag, ValInput)

inputnum = size(TrainInput, 2);
numSum=inputnum*hiddennum+hiddennum;

MAX_GEN = 5;        % 最大迭代次数
POP_SIZE = 50;        % 群体大小
c1 = 0.5;
c2 = 0.5;
w = 0.5;
range = sqrt(6)./sqrt(4+hiddennum);
[x y] = PSO(@(x)FitnessFunc(x,inputnum,hiddennum,TrainInput, PosTag,...
    ValInput, PosValTag), POP_SIZE, c1, c2, w, MAX_GEN, numSum, range,...
    InputWeight_init, HiddenBias_init);
 % 提取  
InputWeight = reshape(x(1:inputnum*hiddennum),hiddennum,inputnum);
HiddenBias = reshape(x(inputnum*hiddennum+1:end),...
    hiddennum,1);
end