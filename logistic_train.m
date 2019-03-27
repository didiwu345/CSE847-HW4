function [weights, iter_tot] = logistic_train(predictor, label, threshold, maxiter, rate)

%%% Outputs
% weights: Updated regression coefficients
% iter_tot: Total number of interations

%%% Inputs
% data: Covariates, X0,X1,...,Xp; n by (p+1)
% label: Labels, Y1,...,Yn; n by 1
% threshold: Threshhold for absolute average change in weights
% maxiter: Maximum iteratiions if threshold not reached
% rate: learning rate

weight_old = repelem(0,58);
weight_new = [repelem(1,10),repelem(0,48)];
iter = 0;

while iter < maxiter || sum(abs(weight_new - weight_old))/58 >= threshold
    
    weight_old = weight_new;
    gradient = 0;
    
    for row = 1:size(predictor,1)
       gradient = gradient + -exp(-weight_old * predictor(row,:)' * label(row))*label(row) .* predictor(row,:) ./ (1 + exp(-label(row) * weight_old * predictor(row,:)')) ;
    end
    
    weight_new = weight_old + rate * -(1/size(predictor,1)) .* gradient;
    iter = iter + 1;
    
end

iter_tot = iter;
weights = weight_new;

end