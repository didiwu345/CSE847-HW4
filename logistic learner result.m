
datax = table2array(data);
datay = table2array(labels);

result_1 = [];
row_nums = [200,500,800,1000,1500,2000];

for num = row_nums
    trainx = [datax(1:num,:)'; repelem(1,num)]';
    trainy = datay(1:num);

    testx = [datax(2001:4601,:)';repelem(1,2601)]';
    testy = datay(2001:4601);

    [train_acc, test_acc,~] = Logistic_implement(trainx,trainy,testx,testy,num);
    result_1 = [result_1; [train_acc, test_acc, num]];
end

plot(result_1(:,3),result_1(:,2))