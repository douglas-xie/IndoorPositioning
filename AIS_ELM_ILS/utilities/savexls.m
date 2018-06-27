
columns = {'ELM', 'AIS-ELM', 'PSO-ELM', 'RBF'};
xlswrite('..\Results\20180611\output(N=1000,hidden=10).xls', columns, 'TrainError', 'A1');
xlswrite('..\Results\20180611\output(N=1000,hidden=10).xls', trainError, 'TrainError', 'A2');
xlswrite('..\Results\20180611\output(N=1000,hidden=10).xls', columns, 'TestError', 'A1');
xlswrite('..\Results\20180611\output(N=1000,hidden=10).xls', testError, 'TestError', 'A2');