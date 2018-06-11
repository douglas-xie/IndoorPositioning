function [PR] = calPR1(NumberOfTag,NumberOfReader,d_RT, N, Para)
PT = Para(1);
Pd0 = Para(2); 
d0 = Para(3);
n = Para(4);
sigma = Para(5);
% N = Para(6);
PR = zeros(NumberOfTag, NumberOfReader, N);

for j = 1:N
    AddGauss=sigma * randn(NumberOfTag, NumberOfReader);
    PR(:,:,j)=PT-(Pd0+10.*n.*log10(d_RT./d0)+AddGauss);
end
end