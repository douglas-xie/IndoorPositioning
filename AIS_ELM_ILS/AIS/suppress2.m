function [AbPop,aff,delta,ts] = suppress2(AbPop,aff,delta,ts,algo_num)

[Ix,Iy] = sort(aff);
AbPop = AbPop(Iy,:);
aff = aff(Iy);
delta = delta(Iy);

D1 = dist(AbPop,AbPop');
aux = triu(D1,1);

if algo_num == 2 || algo_num == 3
    temp = sort(D1);
    ts = min(temp(2,:))+ts*(max(max(D1))-min(temp(2,:)));
%     ts = min(0.2,ts);
end

[Is,Js] = find(aux>0 & aux<ts);
remainder = setdiff([1:1:size(AbPop,1)],Is);

AbPop = AbPop(remainder,:);
aff = aff(remainder);
delta = delta(remainder);