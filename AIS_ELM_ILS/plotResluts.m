figure;
averageError = xlsread('Results\20180611\output(N=100).xls', 'trainError');
x = 1:10;
plot(x,averageError(:,1),'go--','MarkerSize',8);hold on;
plot(x,averageError(:,2),'bd--','MarkerSize',8);
plot(x,averageError(:,3),'rx--','MarkerSize',8);
plot(x,averageError(:,4),'ks--','MarkerSize',8);
xlabel('The signal loss constant(\alpha)');
ylabel('Average positioning error');
legend('ELM(sp = 1.0m)', 'AIS-ELM(sp = 1.0m)', 'PSO-ELM(sp = 1.0m)', ...
    'RBF(sp = 1.0m)');
% title(sprintf('N=%d,sigma=%f', N, sigma));