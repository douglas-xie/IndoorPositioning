function [OutputOfTrain,OutputOfTest,trainEndTime,testEndTime] = ...
    ELM(PosTag,TrainInput, TestInput, NumberofHidden,InputWeight,HiddenBias)
    C = 10;    % regularization coefficient
    NumberofTag = size(TrainInput, 1);
    NumberofTestTag = size(TestInput, 1);
    
    tic;       % train timing start
%% training phase
    tempH = TrainInput * InputWeight';
    ind = ones(1, NumberofTag);
    tempH = tempH + HiddenBias(:,ind)';
    H = 1 ./ (1 + exp(-tempH)); % hidden layer output when using train input
%     H = sin(tempH);
    
    %%%%%% Calculate the output weights by psudo inverse%%%%%%%%%%%%%%%%
    OutputWeight=pinv(H) * PosTag;
    % regularization term
%     OutputWeight = pinv(H'*H + eye(size(H'*H))./C) * H' * PosTag;
    OutputOfTrain = H * OutputWeight;
    trainEndTime = toc; % train timing finish
%% testing phase
    tic;              % test timing start

    tempH1 = TestInput * InputWeight';
    tempH1 = tempH1 + HiddenBias(:, ones(1, NumberofTestTag))';
    H1 = 1./(1+exp(-tempH1));  % hidden layer output when using test input
    OutputOfTest = H1 * OutputWeight;     % prediction of testtag
    testEndTime = toc; % train timing finish
    
end