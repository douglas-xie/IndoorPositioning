figure(1);
testError = xlsread('..\Results\ÓÐÐ§\sp=2.0,N=1000,itears=100.xlsx', 'testError');
averageError = testError;
x = 1:0.5:10;
plot(x,averageError(:,1),'go-','MarkerSize',8);hold on;
plot(x,averageError(:,2),'bd-','MarkerSize',8);
plot(x,averageError(:,3),'rx-','MarkerSize',8);
plot(x,averageError(:,4),'ks-','MarkerSize',8);
xlabel('The signal loss constant(\alpha)');
ylabel('Average positioning error');
legend('ELM(sp = 2.0m)', 'AIS-ELM(sp = 2.0m)', 'PSO-ELM(sp = 2.0m)', ...
    'RBF(sp = 2.0m)');
% title(sprintf('N=%d,sigma=%f', N, sigma));