function [accuracy_train, accuracy_test, iters] = Logistic_implement(trainx,trainy,testx,testy,num)

% accuracy: test accuracy
% size: number of individuals in training set

% optim weights & number of iterations
[opt_w,iter] = logistic_train(trainx, trainy, 0.00001, 5000, 0.001);
iters = iter;
accuracy_train = sum(double(exp(opt_w * trainx')./(1 + exp(opt_w * trainx')) >= 0.5) == trainy')/num;

% test optim weights
accuracy_test = sum(double(exp(opt_w * testx')./(1 + exp(opt_w * testx')) >= 0.5) == testy')/2601;

end
