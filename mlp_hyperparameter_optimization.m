% Learning Hyperparameters optimization
% Momentum Values 
momentum = [0.9 0.7 0.5 0.3 0.1]
momentum_length = length(momentum)

%Learning Rate Values
learning_rate = [0.09 0.07 0.05 0.03 0.01]
lr_length = length(learning_rate)

%Parameter Hypertuning
optimal_cross_entropy_mlp = 100; %declare the initial cross entropy value to be changed within the grid search
optimal_accuracy_mlp = -1; %declare the initial accuracy value to be changed within the grid search
sum_accuracy_mlp = 0; %declare the summary of the accuracies to be changed within the grid search
num_loops_mlp = 0; %declare the number of loops to be changed within the grid search
min_accuracy_mlp = 101; %declare the initial minimum accuracy to be changed within the grid search
%%
%Network Architecture with hidden layer size=100
net_mlp_grid = patternnet(hiddenLayerSize, 'traingdx', 'crossentropy'); % patternnet for classification / traingdx combines adaptive learning rate with momentum training

%%

%Grid Search to find the optimal values of Momentum and Learning Rate 
for i=1:lr_length %learning rate loop
    for j=1:momentum_length %momentum loop
        %Default Network Options
        net_mlp_grid.trainParam.epochs = 1000; %maximum number of epochs
        net_mlp_grid.trainParam.goal = 0;
        net_mlp_grid.trainParam.lr = learning_rate(i); %Learning Rate
        net_mlp_grid.trainParam.max_fail = 7; %Validation vectors are used to stop training early if the network performance 
        %on the validation vectors fails to improve or remains the same for max_fail epochs in a row.
        net_mlp_grid.trainParam.mc = momentum(j); %Momentum
        net_mlp_grid.trainParam.min_grad = 1e-5;
        net_mlp_grid.trainParam.show = 25;
        net_mlp_grid.trainParam.showCommandLine = false;
        net_mlp_grid.trainParam.showWindow = true;
        net_mlp_grid.layers{1}.transferFcn = 'logsig' %log-sigmoid transfer function
        net_mlp_grid.trainParam.time = inf;
        net_mlp_grid.divideParam.trainRatio = 80/100; %divide the dataset in train and validation set, trainRatio=80/100
        net_mlp_grid.divideParam.valRatio = 20/10; % divide the dataset in train and validation set, valRatio=20/100
        net_mlp_grid.divideParam.testRatio = 0/100;
        net_mlp_grid.trainParam.time = inf;
        
        [net_mlp_grid tr_grid Y_grid] = train(net_mlp_grid,X_train,y_train); %train the network
        
        %Predictions for each loop
        y_pred_train_mlp = net_mlp_grid(X_train); %Feed the network with the training data
        [prediction_grid,Labels] = max(y_pred_train_mlp); %keep the maximum probability from the predictions made
        Labels=Labels-1; %Restore the classes to their default values from 0 to 9
        
        perf_train_mlp = perform(net_mlp_grid,y_train,y_pred_train_mlp); %measure the performance of the network
        cross_entropy_train_mlp = crossentropy(y_train,y_pred_train_mlp);%calculate the cross entropy
        accuracy_train_grid_mlp = sum(Labels == y_train_classes)/numel(y_train_classes); %calculate the accuracy
        sum_accuracy_mlp = sum_accuracy_mlp + accuracy_train_grid_mlp;  %calculate summary of all the accuracies
        num_loops_mlp = num_loops_mlp + 1; % calculate the number of loops
        
        if(accuracy_train_grid_mlp > optimal_accuracy_mlp) %Calculate the best performing MLP network based on the accuracy
            optimal_cross_entropy_mlp = cross_entropy_train_mlp; % save the optimal cross entropy if the condition is being satisfied
            optimal_accuracy_mlp = accuracy_train_grid_mlp; % save the maximum accuracy if the condition is being satisfied
            optimal_net_mlp = net_mlp_grid;  % save the optimal MLP net if the condition is being satisfied
            optimal_lr_mlp = learning_rate(i); %save the optimal learning rate if the condition is being satisfied
            optimal_momentum_mlp = momentum(j); %save the optimal momentum if the condition is being satisfied
        end
        if(accuracy_train_grid_mlp<min_accuracy_mlp) %find the minimum accuracy of the grid search
            min_accuracy_mlp = accuracy_train_grid_mlp; 
        end
        disp('Accuracy Grid-search MLP: ' + string(accuracy_train_grid_mlp) + '%')
        disp('Learning Rate Grid-search MLP: ' + string(learning_rate(i)))
        disp('Momentum Grid-search MLP: ' + string(momentum(j)))
    end
end

avg_accuracy_mlp = sum_accuracy_mlp/num_loops_mlp; %calcylate the average accuracy of the grid search 
%%
disp('Optimal Accuracy MLP Grid Search: ' + string(optimal_accuracy_mlp*100) + '%') %Print the Optimal MLP Accuracy of the Grid Search
disp('Optimal Learning Rate MLP Grid Search: ' + string(optimal_lr_mlp))  %Print the Optimal Learning Rate of the Grid Search
disp('Optimal Momentum MLP Grid Search: ' + string(optimal_momentum_mlp))  %Print the Optimal Momentum of the Grid Search
disp('Average Accuracy MLP Grid Search: ' + string(avg_accuracy_mlp*100) +'%') %Print the Average Accuracy of the Grid Search
disp('Minimum Accuracy MLP Grid Search: ' + string(min_accuracy_mlp*100) +'%')  %Print the Minimum Accuracy of the Grid Search
