function aff = affinity(AbPop, NumberofHidden, NumberofTag, TrainInput,...
    PosTag, NumberofValidation, ValidationInput, PosValidation)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 计算AIS算法种群每个个体的亲和力，AbPop 为需要计算的种群，返回aff 种群亲和力，
% aff 值越大，所代表的个体更优。
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[row, col] = size(AbPop);
NumberofInput = size(TrainInput, 2);

aff = zeros(row, 1);
for irow = 1:row
   InputWeight = reshape(AbPop(irow,1:NumberofHidden*NumberofInput)', ...
     NumberofHidden, NumberofInput);
   HiddenBias = reshape(AbPop(irow,NumberofHidden*NumberofInput+1:end)',...
       NumberofHidden, 1);  
   % 使用ELM估计坐标值，用以计算误差
   Output=ELM(PosTag,TrainInput, ValidationInput, NumberofHidden,InputWeight,HiddenBias);

    Error = calLoss(NumberofValidation, Output, PosValidation);
    aff(irow, 1) = 1./(1+Error);
end