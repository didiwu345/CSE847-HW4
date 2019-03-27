function[weight,bias,feature_selected] = logistic_l1_train(data, labels, par)

opts.rFlag = 1;
opts.tol = 1e-6;
opts.tFlag = 4;
opts.maxIter = 5000;

[weight,bias] = LogisticR(data, labels, par, opts);

feature_selected = 115 - sum(double(weight(1:115) == 0));


end