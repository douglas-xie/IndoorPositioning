function  [PR1, PRVal1, PRTest1] = addDeltaPR(PR, PRVal, PRTest)
NumberofRefTag = size(PR, 1);
NumberofValTag = size(PRVal, 1);
NumberofTestTag = size(PRTest, 1);

PRGroup = [PR; PRVal; PRTest];
PRGroup(:, 5) = PRGroup(:, 2) - PRGroup(:, 1);
PRGroup(:, 6) = PRGroup(:, 3) - PRGroup(:, 1);
PRGroup(:, 7) = PRGroup(:, 4) - PRGroup(:, 1);
% PRGroup(:, 8) = PRGroup(:, 1) - PRGroup(:, 2);
% PRGroup(:, 9) = PRGroup(:, 3) - PRGroup(:, 2);
% PRGroup(:, 10) = PRGroup(:, 4) - PRGroup(:, 2);
% PRGroup(:, 11) = PRGroup(:, 1) - PRGroup(:, 3);
% PRGroup(:, 12) = PRGroup(:, 2) - PRGroup(:, 3);
% PRGroup(:, 13) = PRGroup(:, 4) - PRGroup(:, 3);
% PRGroup(:, 14) = PRGroup(:, 1) - PRGroup(:, 4);
% PRGroup(:, 15) = PRGroup(:, 2) - PRGroup(:, 4);
% PRGroup(:, 16) = PRGroup(:, 3) - PRGroup(:, 4);

PR1 = PRGroup(1:NumberofRefTag, :);
PRVal1 = PRGroup(NumberofRefTag + 1:NumberofRefTag + NumberofValTag, :);
PRTest1 = PRGroup(NumberofRefTag + NumberofValTag + 1:end, :);

end