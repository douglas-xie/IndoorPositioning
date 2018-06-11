function [best_ab,best_fval,it,best_set,FE,emin,emax,emean,Emin,Emax,Emean,estd,Estd]=optainet(center,N_cluster,AmtTtag,PRdeltaPRGT,PosTtag,AmtVtag,PRdeltaPRGV,PosVtag,Amttesttag,PRtestdeltaPRtestG,Postesttag)
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global dimension;
global pop_size lb ub max_it;
dimension = 4;
pop_size = 50;
max_it= 5;
lb = -10;%中心向量取值下限%
ub = 0; %中心向量取值上限%
FE = 0;
%AbPop = reshape(center',1,N_cluster*7);
AbPop = rand(pop_size,N_cluster*dimension).*(ub-lb)+lb; % Initialize the population
aff = affinity(AbPop,N_cluster,AmtTtag,PRdeltaPRGT,PosTtag,AmtVtag,PRdeltaPRGV,PosVtag);   % Evaluate the affinity
FE = FE+pop_size;
aff_av_old = mean(aff); % Evaluate the average affinity
best_fval = -max(aff);
best_set = best_fval;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

it = 0;
ts = 0.2*2^0.5*(ub-lb);
while (it < max_it)
    it
    % Cloning, mutation and selection
    [AbPop,aff,FE] = clone_mut_select4optainet(AbPop,aff,lb,ub,...
     N_cluster,AmtTtag,PRdeltaPRGT,PosTtag,AmtVtag,PRdeltaPRGV,PosVtag,FE);
    aff_av = mean(aff);
    % Suppression and recuitment
    min_aff = min(aff);
    if abs(1-(aff_av_old-min_aff)/(aff_av-min_aff)) < 0.1
        N_sup_old = size(AbPop,1);
        [AbPop,aff] = suppress(AbPop,aff,ts,1);
        N_sup = size(AbPop,1);
        if N_sup < N_sup_old
            d = round(0.4*pop_size);
            d = min(d,150-N_sup);
            Ab1 = lb + rand(d,dimension).*(ub-lb);
            AbPop = [AbPop;Ab1];
            aff_new = affinity(AbPop,N_cluster,AmtTtag,PRdeltaPRGT,PosTtag,AmtVtag,PRdeltaPRGV,PosVtag);
            aff = [aff;aff_new];
            clear Ab1;
        end
        aff_av = mean(aff);
    end
    aff_av_old = aff_av;
    it = it+1;
    [Ix,Iy] = max(aff);
    best_ab = AbPop(Iy,:);
    best_fval = -Ix;
    best_set = [best_set;best_fval];
    center = reshape(best_ab,dimension,N_cluster)';
    for i = 1:N_cluster
        eucenter = zeros(N_cluster, 1);
        for j = 1:N_cluster
            eucenter(j, 1) = norm(center(i,:) - center(j,:));
        end
        rbfvar(i, 1) = mean(eucenter);
    end

% %--------------------------------------------通过伪逆确定权值-------------------------------------------%
for i = 1:N_cluster
    for k = 1:AmtTtag
        G(k,i) = exp((-1/(2*rbfvar(i,1)^2))*norm(PRdeltaPRGT(k,:)-center(i,:)).^2);
    end
end
rbfweight = pinv(G)*(PosTtag);
%%-----------------------------------------------------------------------------%
for k = 1:AmtVtag
    tempy1=0;
    tempy2=0;
    for i = 1:N_cluster
        tempy1 = rbfweight(i,1).*exp((-1/(2*rbfvar(i,1)^2))*norm(PRdeltaPRGV(k,:)-center(i,:)).^2)+tempy1;
        tempy2 = rbfweight(i,2).*exp((-1/(2*rbfvar(i,1)^2))*norm(PRdeltaPRGV(k,:)-center(i,:)).^2)+tempy2;
    end
    y(k,1)= tempy1;
    y(k,2)= tempy2;
end
eutemp=0;
for i=1:AmtVtag
    eutemp = eutemp + norm(PosVtag(i,:)-y(i,:));
end

error = eutemp./AmtVtag

errorv(it,1)=error;
emin(it,1)=min(errorv(:,1));
emax(it,1)=max(errorv(:,1));
emean(it,1)=mean(errorv(:,1));
estd(it,1)=std(errorv(:,1));

% %---------------------------------------训练完毕使用神经网络------------------------------------------%
for k = 1:Amttesttag
    tempy1=0;
    tempy2=0;
    for i = 1:N_cluster
        tempy1 = rbfweight(i,1).*exp((-1/(2*rbfvar(i,1)^2))*norm(PRtestdeltaPRtestG(k,:)-center(i,:)).^2)+tempy1;
        tempy2 = rbfweight(i,2).*exp((-1/(2*rbfvar(i,1)^2))*norm(PRtestdeltaPRtestG(k,:)-center(i,:)).^2)+tempy2;
    end
    y(k,1)= tempy1;
    y(k,2)= tempy2;
end
%--------------------------------计算定位误差------------------------------------------------%
eutemp=0;
for i=1:Amttesttag
    eutemp = eutemp + norm(Postesttag(i,:)-y(i,:));
end
eutemp./Amttesttag 

Error(it,1) = eutemp./Amttesttag ;
Emin(it,1)=min(Error(:,1));
Emax(it,1)=max(Error(:,1));
Emean(it,1)=mean(Error(:,1));
Estd(it,1)=std(Error(:,1));

    % Display
    %disp(sprintf('It: %d Best: %f Av: %f Net size: %d FE: %d',it,log(best_fval)/log(10),mean(-aff),size(AbPop,1),FE));
end