% CNN Optimal Architecture 
layers = [
    imageInputLayer([28 28 1]) % Input grayscale image 28x28
    
    convolution2dLayer(3,4,'Padding','same') % First Convolution Layer (filter_size, num_of_filters, Name, Value)
    batchNormalizationLayer % batch normalization normalizing the outputs of the convulation that helps converge with higher learning rates
    reluLayer % Activation function
    
    maxPooling2dLayer(2,'Stride',2) %POOL layer will perform a downsampling operation along the spatial dimensions (width, height), resulting in volume 
   
    convolution2dLayer(3,8,'Padding','same') % Second Convolution Layer (filter_size, num_of_filters, Name, Value)
    batchNormalizationLayer % Batch Normalization to increase the training speed of the model
    reluLayer % Activation function
    
    fullyConnectedLayer(10) %Fully connected layer with output size = 10
    softmaxLayer
    classificationLayer];
%% 
%Learning Hyperparameters optimization

%Momentum Values 
momentum = [0.9 0.7 0.5 0.3 0.1];
momentum_length = length(momentum); %length of Momentum vector for later use

%Learning Rate Values
learning_rate = [0.09 0.07 0.05 0.03 0.01]; %length of Learning Rate vector for later use
lr_length = length(learning_rate);

%Implement GridSearch based on the Optimal architecture found from the experiments previously
optimal_accuracy_cnn = -1; %Declare Accuracy
min_accuracy = 101; % Declare minimum accuracy
sum_accuracy = 0; % Summary of accuracy for the average accuracy
num_loops = 0; %Number of loops for the average accuracy
for i=1:lr_length
    for j=1:momentum_length
        % Specify training options and Hyperparameters
        options = trainingOptions('sgdm', ... % stochastic gradient descent with momentum
            'InitialLearnRate', learning_rate(i), ... % changing learning rate
            'Momentum', momentum(j), ... % changing momentum
            'MaxEpochs',7, ... % maximum epochs
            'Shuffle','every-epoch', ... % shuffle the data in every epoch
            'ValidationData',validation_imds, ... % use the validation set to make predictions
            'ValidationFrequency',30, ...
            'Verbose',false, ...
            'Plots','None');
        
        net = trainNetwork(train_imds, layers, options);  %Train network for this loop
    
        Y_predicted_val = classify(net,validation_imds); %Calculate predictions of this net
        Y_val = validation_imds.Labels; %Keep the labels of the validation images
    
        accuracy_validation_set = sum(Y_predicted_val == Y_val)/numel(Y_val); % Calculate the accuracy
        sum_accuracy = sum_accuracy + accuracy_validation_set; % add accuracy value to the sum
        num_loops = num_loops +1; % add 1 loop to the number of loops
        
        if(accuracy_validation_set > optimal_accuracy_cnn) %Calculate the best performing network based on the accuracy
            optimal_accuracy_cnn = accuracy_validation_set; % save the maximum accuracy if the condition is being satisfied
            optimal_net_cnn = net; % save the optimal net if the condition is being satisfied
            optimal_options_cnn = options; % save the optimal options if the condition is being satisfied
            optimal_lr_cnn = learning_rate(i); %save the optimal learning rate if the condition is being satisfied
            optimal_momentum_cnn = momentum(j); %save the optimal momentum if the condition is being satisfied
        end
        if(accuracy_validation_set< min_accuracy) %Find the minimum accuracy
            min_accuracy = accuracy_validation_set; % save the minimum accuracy if the condition is satisfied
        end    
        disp('Accuracy validation: ' + string(accuracy_validation_set)) %Print the accuracy for each loop for testing purposes
        disp('Learning Rate: ' + string(learning_rate(i))) %Print the learning rate for each loop for testing purposes
        disp('Momentum: ' + string(momentum(j))) %Print the momentum for each loop for testing purposes
    end
end

avg_accuracy = (sum_accuracy*100)/num_loops; %Calculate the average accuracy of the Grid Search loops
%%
disp('Optimal Accuracy CNN Grid Search: ' + string(optimal_accuracy_cnn*100) + '%') %Print the Optimal Accuracy of the Grid Search
disp('Optimal Learning Rate CNN Grid Search: ' + string(optimal_lr_cnn)) %Print the Optimal Learning Rate of the Grid Search
disp('Optimal Momentum CNN Grid Search: ' + string(optimal_momentum_cnn)) %Print the Optimal Momentum of the Grid Search
disp('Average accuracy CNN Grid Search: ' + string(avg_accuracy) +'%') %Print the Average Accuracy of the Grid Search
disp('Minimum accuracy CNN Grid Search: ' + string(min_accuracy*100) +'%') %Print the Minimum Accuracy of the Grid Search

