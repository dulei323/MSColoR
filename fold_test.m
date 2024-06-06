function res = fold_test(data, para)

Kfold =5;  
[g, c] = size(data.Y);
n = size(data.X{1},1);

fprintf('===================================\n');
indices = crossvalind('Kfold', n, Kfold);
for k = 1 : Kfold
    fprintf('Current fold: %d\n', k);
    test = (indices == k);
    train = ~test;
    
    % Split training data and test data, and normalization
    for i = 1:g
        trainData.X{i} = data.X{i}(train, :);
        testData.X{i} = data.X{i}(test, :);
        for j = 1 : c
        testData.Y{i, j} = Normalization(data.Y{i, j}(test, :));
        trainData.Y{i, j} = Normalization(data.Y{i, j}(train, :));
        end 
    end
    
    % Training step
    [W(k, :), V(k, :), b(k, :), A(k, :), B(k, :)] = MSColoR(trainData.X, data.X_ref, trainData.Y, data.beta, para, data.t);                                                                                     
   
    % RMSE 
    for i = 1:g  
        res.rmse.train(k, i)=  eval_res(trainData.X{i}, trainData.Y(i, :), data.t(i, :), W{k, i}, V{k, i}, A(k, :), B(k, :));
        res.rmse.test(k, i)=  eval_res(testData.X{i}, testData.Y(i, :), data.t(i, :), W{k, i}, V{k, i}, A(k, :), B(k, :));
    end 
end

%% Weights

res.W = W(1, :);  res.V = V(1, :); res.b = b(1, :);
for i = 1:g
    for j = 2:Kfold
        res.W{i} = res.W{i} + W{j, i}; 
        res.V{i} = res.V{i} + V{j, i};
        res.b{i} = res.b{i} + b{j, i};
    end
    res.W{i} = res.W{i} / Kfold; res.V{i} = res.V{i} / Kfold; res.b{i} = res.b{i} / Kfold;
end
res.A = mean(A); res.B = mean(B);

%% RMSE 

res.rmse_train = [mean(res.rmse.train)'  std(abs(res.rmse.train))']; 
res.rmse_test = [mean(res.rmse.test)' std(abs(res.rmse.test))'];