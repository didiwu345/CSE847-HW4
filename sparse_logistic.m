pars = [0,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1];

result = [];

for par=pars
    
    [optim_w, optim_bias, features] = logistic_l1_train(X_train,y_train,par);
    % probabilities for y=1, in training set
    prob_train = exp(X_train * optim_w + optim_bias) ./ (1 + exp(X_train * optim_w + optim_bias));
    train_result = 2*(double(prob_train >= 0.5)) - 1;
    train_accuracy = sum(double(train_result == y_train))/size(y_train,1);
    
    % probabilitis for y=1, in testing set
    prob_test = exp(X_test * optim_w + optim_bias) ./ (1 + exp(X_test * optim_w + optim_bias));
    test_result = 2*(double(prob_test >= 0.5)) - 1;
    test_accuracy = sum(double(test_result == y_test))/size(y_test,1);
    
    result = [result;[train_accuracy, test_accuracy, features, par]];
    
end

%%% ROC
[X,Y] = perfcurve(y_test, prob_test,1);
plot(X,Y)