%Import the Data from the created directories
%Train data
train_fashion_dataset_path = fullfile('..', 'Konstantinos_Gkolias_NC_coursework', 'train') % train dataset path
train_imds = imageDatastore(train_fashion_dataset_path, ...
    'IncludeSubfolders',true,'LabelSource','foldernames'); %save the generated images in an Image Datastore

train_imds_full = train_imds; %keep the original train dataset before splitting it, to train the final model

%Split Train and Validation Datasets 80% Train and 20% Validation based on
%the labels
[train_imds,validation_imds] = splitEachLabel(train_imds,0.8);

%Test data
test_fashion_dataset_path = fullfile('..', 'Konstantinos_Gkolias_NC_coursework','test')
test_imds = imageDatastore(test_fashion_dataset_path, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames')
%% 
%Display examples of the train images randomly
figure;
perm = randperm(48000,30);
for i = 1:30
    subplot(5,6,i);
    imshow(train_imds.Files{perm(i)});
end

%% Check the size of the training, validation and testing sets
trainFull_labelCount = countEachLabel(train_imds_full) %count the labels of the initial train set
train_labelCount = countEachLabel(train_imds) %count the labels of the 80% train set, without the holdout validation set
validation_labelCount = countEachLabel(validation_imds) %count the labels of the validation set
test_labelCount = countEachLabel(test_imds) %count the labels of the test set

%% CNN Baseline Mode
% CNN Layers Initialization and Architecture for Baseline model with Default
% parameters
%Source: https://uk.mathworks.com/help/deeplearning/examples/create-simple-deep-learning-network-for-classification.html

% After testing the performance of different architectures using the default network options, we used 3
% Convolutional Layers with 1 Maxpooling Layer Between them and a fully
% connected layer.
layers = [
    imageInputLayer([28 28 1]) % Input grayscale image 28x28 
    
    convolution2dLayer(3,4,'Padding','same') % First Convolution Layer (filter_size, num_of_filters, Name, Value)
    batchNormalizationLayer % Batch Normalization to increase the training speed of the model
    reluLayer % Activation function Introduces non-linearity to the model
    
    maxPooling2dLayer(2,'Stride',2) % Pool layer will perform a downsampling operation along the spatial dimensions (width, height)
   
    convolution2dLayer(3,8,'Padding','same') % Second Convolution Layer (filter_size, num_of_filters, Name, Value)
    batchNormalizationLayer % Batch Normalization to increase the training speed of the model
    reluLayer % Introduces non-linearity to the model
    
    fullyConnectedLayer(10) %Fully connected layer with output size = 10
    softmaxLayer %softmax layer
    classificationLayer];

%Specify training options and Hyperparameters
options = trainingOptions('sgdm', ... %stochastic gradient descent with momentum
    'InitialLearnRate',0.01, ... %Learning Rate
    'MaxEpochs',7, ... % maximum epochs
    'Shuffle','every-epoch', ... %shuffle the data in every epoch
    'ValidationData',validation_imds, ... %use the validation set as validation data
    'ValidationFrequency',30, ... %specify the validation frequency
    'Verbose',false, ...
    'Plots','training-progress'); %plot in the training phase

%Train the Network for Baseline Model based on the layers and options
%declared
net_baseline = trainNetwork(train_imds, layers, options);

%% VALIDATION in the Baseline Model

%Validation Images Classification and Accuracy Computation
Y_predicted_validation = classify(net_baseline,validation_imds); %Make predictions from the trained baseline network
Y_Validation = validation_imds.Labels; % Keep the labels of the images

%Accuracy for Baseline Model in Validation Set
accuracy_validation_set = sum(Y_predicted_validation == Y_Validation)/numel(Y_Validation);

disp('The accuracy of the baseline validation set is: ' + string(accuracy_validation_set*100) +'%')

%% Grid Search CNN: Optimal Architecture and Hyperparameters Optimization
tic
run('cnn_hyperparameter_optimization.m')
toc

%% Train the model with the Best Architecture and the Best Hyperparameters based on the Grid Search

layers = [
    imageInputLayer([28 28 1]) %input layer
    
    
    convolution2dLayer(3,4,'Padding','same') % first convolutional layer (filter_size, num_of_filters, Name, Value)
    batchNormalizationLayer % batch normalization normalizing the outputs of the convulation that helps converge with higher learning rates
    reluLayer %relu Layer
    
    maxPooling2dLayer(2,'Stride',2) %POOL layer will perform a downsampling operation along the spatial dimensions (width, height), resulting in volume [16x16x12].
   
    convolution2dLayer(3,8,'Padding','same') % second convolutional layer (filter_size, num_of_filters, Name, Value)
    batchNormalizationLayer  % batch normalization normalizing the outputs of the convulation that helps converge with higher learning rates
    reluLayer
    
    fullyConnectedLayer(10) %fully connected layer
    softmaxLayer %softmax layer
    classificationLayer];

options = trainingOptions('sgdm', ...
            'InitialLearnRate', optimal_lr_cnn, ... %take the optimal learning rate value after grid search
            'Momentum', optimal_momentum_cnn, ... %take the optimal momentum value after grid search
            'MaxEpochs',7, ... %maximum number of 7 epochs used for early stopping to prevent overfitting
            'Shuffle','every-epoch', ... %apply shuffling in every epoch to prevent overfitting
            'ValidationData',validation_imds, ... % use the validation data for early stopping to prevent overfitting
            'ValidationFrequency',30, ...  %specify the validation frequency
            'Verbose',false, ...
            'Plots','training-progress');


%Train the Network with the train data
final_cnn_net = trainNetwork(train_imds, layers, options)

%%  Results of the trained MLP network on Train Data, for comparison with the Final Model on test data

y_predicted_train_optimal = classify(final_cnn_net,train_imds); %predictions of the optimal network based on the train data
y_train_optimal = train_imds.Labels; % class labels of the full train dataset

%Accuracy for Optimal Model on Train Set after Hyperparameter Tuning
accuracy_train_set = sum(y_predicted_train_optimal == y_train_optimal)/numel(y_train_optimal);

disp('Accuracy of optimal CNN model after hyperparameter tuning for the train data: ' + string(accuracy_train_set*100) +'%')
%% Results of the trained network on the unseen Test data

%Test Images Classification and Accuracy Computation
Y_predicted_test_optimal = classify(final_cnn_net,test_imds);  %predictions of the optimal network based on the test data
Y_test_optimal = test_imds.Labels;  % class labels of the test dataset

%Accuracy for Optimal Model on Test Set after Hyperparameter Tuning
accuracy_test_set = sum(Y_predicted_test_optimal == Y_test_optimal)/numel(Y_test_optimal);

disp('Accuracy of optimal model after hyperparameter tuning on unseen test data: ' + string(accuracy_test_set*100) +'%')
%%  Plot Confusion Matrix CNN for Test set
plotconfusion(Y_test_optimal,Y_predicted_test_optimal)

%% Print Final Values of Accuracies in different Stages CNN
disp('The accuracy of the baseline validation set is: ' + string(accuracy_validation_set*100) +'%')
disp('Optimal Accuracy CNN Grid Search: ' + string(optimal_accuracy_cnn*100) + '%') %Print the Optimal Accuracy of the Grid Search
disp('Optimal Learning Rate CNN Grid Search: ' + string(optimal_lr_cnn)) %Print the Optimal Learning Rate of the Grid Search
disp('Optimal Momentum CNN Grid Search: ' + string(optimal_momentum_cnn)) %Print the Optimal Momentum of the Grid Search
disp('Average accuracy CNN Grid Search: ' + string(avg_accuracy) +'%') %Print the Average Accuracy of the Grid Search
disp('Minimum accuracy CNN Grid Search: ' + string(min_accuracy*100) +'%') %Print the Minimum Accuracy of the Grid Search
disp('Accuracy of optimal CNN model after hyperparameter tuning for the train data: ' + string(accuracy_train_set*100) +'%') %Print the optimal model's accuracy on train set
disp('Accuracy of optimal model after hyperparameter tuning on unseen test data: ' + string(accuracy_test_set*100) +'%') %Print the optimal model's accuracy on test set
%% MLP IMPLEMENTATION 

run('mlp_implementation.m') %run the MLP implementation script
