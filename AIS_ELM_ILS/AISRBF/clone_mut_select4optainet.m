function [C,Out,FE] = clone_mut_select4optainet(AbPop,aff,lb,ub,N_cluster,AmtTtag,PRdeltaPRGT,PosTtag,AmtVtag,PRdeltaPRGV,PosVtag,FE)

[N,L] = size(AbPop);
C = [];
Out = [];

% Normalization
aff_norm = norma(aff);

beta=0.8;%变异的步长，现在是增加或者减少0.01;那么至少100次才能到达1.
Nc=100;

for i=1:N
   % Clone
   vones = ones(Nc,1);
   Cc = vones * AbPop(i,:);
   % Mutation 
   g =normrnd(0,1,Nc,L).* exp(-aff_norm(i)./beta);
 
   g(1,:) = zeros(1,L);	% Keep one previous individual for each clone unmutated
   c = Cc + g;
   % Keeps all elements of the population within the allowed bounds
   for j=1:L
        Ixmin=find(c(:,j) < lb);
        Ixmax=find(c(:,j) > ub);
        if ~isempty(Ixmin)
            c(Ixmin,j) = Cc(length(Ixmin),j);
        end;
        if ~isempty(Ixmax)
            c(Ixmax,j) = Cc(length(Ixmax),j);
        end;
    end
   % Evaluate the fitness
   aff = affinity(c,N_cluster,AmtTtag,PRdeltaPRGT,PosTtag,AmtVtag,PRdeltaPRGV,PosVtag);
   FE = FE+Nc-1;
   [out,I] = max(aff);
   C = [C;c(I,:)];  % C contains only the best individuals of each clone
   Out=[Out;out];
end;