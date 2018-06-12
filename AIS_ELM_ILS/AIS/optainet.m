function [best_ab,best_fval,it,best_set,FE]=...
    optainet(InputWeight,HiddenBias,NumberofHidden,...
    NumberofTag, PosTag,TrainInput, NumberofValidation, ...
    PosValidation, ValidationInput)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The opt-aiNet optimization
% func_num: the number of the benchmark function
% dimension:    the dimension of the antibody
% best_ab:  the antibody with the highest affinity
% best_fval:    the function value of best_ab
% it:   the iterations cost when terminated
% best_set:  the set of best function value in each iteration
% FE:   the number of the function evaluation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global pop_size lb ub max_it;
pop_size = 50;
max_it = 5;
NumberofInput = size(InputWeight, 2);
lb = -sqrt(6)./sqrt(NumberofInput+NumberofHidden); % 输入权重隐层神经元偏置取值上限
ub = sqrt(6)./sqrt(NumberofInput+NumberofHidden);
FE = 0;

% Initialize the population
AbPop = rand(pop_size, NumberofHidden*(1+NumberofInput)) .* (ub-lb) + lb;
AbPop(1,:) = [InputWeight(:); HiddenBias(:)]';
% Evaluate the affinity
aff = affinity(AbPop, NumberofHidden, NumberofTag, TrainInput, PosTag,...
    NumberofValidation, ValidationInput, PosValidation); 
FE = FE + pop_size;
aff_av_old = mean(aff);    % Evaluate the average affinity
best_fval = max(aff);% Select the max aff corresponding to the min error
best_set = best_fval;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

it = 1;
ts = 0.2*2^0.5*(ub-lb);
best_fval_old = 0;
while(abs(best_fval-best_fval_old)>1e-15 || it <= max_it)
%    fprintf('The generation is:%d\n',it);
   % Cloning, mutation and selection
   best_fval_old = best_fval;
   [AbPop, aff, FE] = clone_mut_select4optainet(AbPop, aff, lb, ub, ...
       NumberofHidden, NumberofTag, TrainInput, PosTag, ...
       NumberofValidation, ValidationInput, PosValidation,FE);   
   aff_av = mean(aff);
   % Suppression and recuitment
   min_aff = min(aff);
   if abs(1-(aff_av_old-min_aff)/(aff_av-min_aff)) < 0.1   % 检查相似度
       N_sup_old = size(AbPop, 1);
       [AbPop,aff] = suppress(AbPop,aff,ts,2);  
       N_sup = size(AbPop,1);
       if N_sup < N_sup_old  % 有被抑制的抗体
          d = round(0.4*pop_size);
          d = min(d,150-N_sup);         
          Ab1 = lb + rand(d,size(AbPop, 2)) .* (ub-lb); 
          AbPop = [AbPop;Ab1];
          aff_new=affinity(AbPop, NumberofHidden, NumberofTag, TrainInput,...
              PosTag,NumberofValidation, ValidationInput, PosValidation);
          aff = [aff;aff_new];
          clear Ab1;
       end
       aff_av = mean(aff);
   end
   aff_av_old = aff_av;
   it = it + 1;
   [Ix, Iy] = max(aff);
   best_ab = AbPop(Iy, :);
   best_fval = Ix;
   best_set = [best_set;best_fval];
end