function [PRFilter] = GaussianFilter(PR, NumberOfTag, NumberOfReader, N)

PRmean=mean(PR,3);  % 测试标签 RSSI 均值
PRd_square = zeros(NumberOfTag, NumberOfReader); % 估计标准差
for i = 1:N
    PRd_square = PRd_square + (PR(:,:,i)-PRmean).^2;
end
sigma = sqrt(1/(N-1) * PRd_square);
%-----筛选出1个标准差范围内的点-----------%
PRuplimit = PRmean+2*sigma;
PRdownlimit = PRmean-2*sigma;

PRTemp=zeros(NumberOfTag,NumberOfReader);
PRTemp1=zeros(NumberOfTag,NumberOfReader);
PRFilter=zeros(NumberOfTag,NumberOfReader);
PRmean1 = zeros(NumberOfTag, NumberOfReader);
PRsigma1 = zeros(NumberOfTag, NumberOfReader);
for i = 1:NumberOfTag
  for j=1:NumberOfReader
    Length=0;
    Val_ind = [];
    for k=1:N
      if abs(PR(i,j,k)-PRmean(i,j)) < 2*sigma(i,j)
         Val_ind = [Val_ind, k];
         PRTemp(i,j)= PRTemp(i,j)+PR(i,j,k);
         Length=Length+1;
      end
    end
    PRmean1(i,j)=PRTemp(i,j)./Length;  % 测试标签滤波输出
    temp = 0;
    for l = 1:Length
        temp = temp + (PR(i,j,Val_ind(1,l)) - PRmean1(i, j)).^2; 
    end
    PRsigma1(i,j) = sqrt(1./(Length-1) * temp);
  end  
end
PRFilter = PRmean1;
% PRuplimit1 = sqrt(2*PRsigma1.*log(6.3*PRsigma1))+PRmean1;
% PRdownlimit1 = sqrt(2*PRsigma1.*log(3.8*PRsigma1))+PRmean1;
% 
% for i = 1:NumberOfTag
%     for j=1:NumberOfReader
%         Length1=0;
%         for k=1:N
%           if PR(i,j,k)<PRuplimit1(i,j) && PR(i,j,k)>PRdownlimit1(i,j)
%              PRTemp1(i,j)= PRTemp1(i,j)+PR(i,j,k);
%              Length1=Length1+1;
%           end
%         end
%         PRFilter(i,j)=PRTemp1(i,j)./Length1;  % 测试标签滤波输出
%     end
% end