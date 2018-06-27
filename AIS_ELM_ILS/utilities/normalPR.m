function [output, PRMIN, PRMAX] = normalPR(PR, PRMIN, PRMAX)
%==========================================================================
% 功能：RSSI 值归一化操作，使其投影到[0,1]。
% 参数：PR - RSSI值
%       PRMIN - RSSI最小值。当主函数前面已计算了最小值时代入。
%       PRMAX - RSSI最大值。
% 返回值：output - RSSI归一化输出，介于0至1之间。
%       PRMIN - RSSI最小值。
%       PRMAX - RSSI最大值。
% 日期：20180605
%==========================================================================
if nargin < 3
    PRMAX = max(PR, [], 1);
    PRMIN = min(PR, [], 1);
end

[irow, jcol] = size(PR);
if length(PRMAX) ~= jcol || length(PRMIN) ~= jcol
   PRMAX = PRMAX(1,1) * ones(1, jcol); 
   PRMIN = PRMIN(1,1) * ones(1, jcol); 
end

PRMAX = ones(irow, 1) * PRMAX;
PRMIN = ones(irow, 1) * PRMIN;

output = (PR - PRMIN)./(PRMAX - PRMIN);
end