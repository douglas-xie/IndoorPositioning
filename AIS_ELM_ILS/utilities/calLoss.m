function [mean_error, max_error, min_error] = calLoss(NumberOfTag,output, y)

error = zeros(NumberOfTag, 1);
for i = 1:NumberOfTag
   error(i, 1) = norm(y(i,:) - output(i,:)); 
end
mean_error = mean(error(:, 1));
max_error = max(error);
min_error = min(error);
% mean_error = mse(y' - output');
end