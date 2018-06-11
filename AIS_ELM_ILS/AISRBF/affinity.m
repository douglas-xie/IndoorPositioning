function aff  = affinity(AbPop,N_cluster,AmtTtag,PRdeltaPRGT,PosTtag,AmtVtag,PRdeltaPRGV,PosVtag)
global dimension;
[row,col]=size(AbPop);
for irow=1:row
    center = reshape(AbPop(irow,1:N_cluster*dimension),dimension,N_cluster)';
 for i = 1:N_cluster
    eucenter = zeros(N_cluster, 1);
    for j = 1:N_cluster
        eucenter(j, 1) = norm(center(i,:) - center(j,:));
    end
    rbfvar(i, 2) = mean(eucenter);
 end

%%%%%%%%%%%%%%%%通过伪逆确定权值%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:N_cluster
    for k = 1:AmtTtag
        G(k,i) = exp((-1/(2*rbfvar(i,2)^2))*norm(PRdeltaPRGT(k,:)-center(i,:)).^2);
    end
end
rbfweight = pinv(G)*(PosTtag);

% %---------------------------------------训练完毕使用神经网络------------------------------------------%
for k = 1:AmtVtag
    tempy1=0;
    tempy2=0;
    for i = 1:N_cluster
        tempy1 = rbfweight(i,1).*exp((-1/(2*rbfvar(i,2)^2))*...
            norm(PRdeltaPRGV(k,:)-center(i,:)).^2)+tempy1;
        tempy2 = rbfweight(i,2).*exp((-1/(2*rbfvar(i,2)^2))*...
            norm(PRdeltaPRGV(k,:)-center(i,:)).^2)+tempy2;
    end
    y(k,1)= tempy1;
    y(k,2)= tempy2;
end
%--------------------------------计算定位误差------------------------------------------------%
eutemp=0;
for i=1:AmtVtag
    eutemp = eutemp + norm(PosVtag(i,:)-y(i,:));
end
error = eutemp./AmtVtag;
aff(irow,1)= -error;
end
end

