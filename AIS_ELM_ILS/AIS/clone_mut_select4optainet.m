function [C, Out, FE] = clone_mut_select4optainet(AbPop, aff, lb, ub, ...
       NumberOfHidden, NumberOfTag, TrainInput, PosTag, ...
       NumberOfValidation, ValidationInput, PosValidation,FE)
[N,L] = size(AbPop);
C = [];
Out = [];

% Normalization
aff_norm = norma((1-aff)./aff);
% aff_norm = norma(aff);

beta = 0.7; % ±‰“Ï≤Ω≥§
Nc = 100;

for i = 1:N
   % Clone
   vones = ones(Nc, 1);
   Cc = vones * AbPop(i,:);
   % Mutation
   g = normrnd(0,1,Nc,L) .* exp(-aff_norm(i)./beta);
   
   g(1,:) = zeros(1, L);
   c = Cc + g;
   
   for j = 1:L
       Ixmin = find(c(:,j) < lb);
       Ixmax = find(c(:,j) > ub);
       if ~isempty(Ixmin)
          c(Ixmin, j) = Cc(length(Ixmin), j); 
       end
       if ~isempty(Ixmax)
          c(Ixmax, j) = Cc(length(Ixmax), j); 
       end
   end
   % Evaluate the fitness
   aff = affinity(c, NumberOfHidden, NumberOfTag, TrainInput, PosTag, ...
       NumberOfValidation, ValidationInput, PosValidation);
   FE = FE+Nc-1;
   [out, I] = max(aff);
   C = [C;c(I,:)];  % C contains only the best individuals of each clone
   Out = [Out;out];
end
   
end