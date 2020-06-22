%Read data
train_data = readtable('fashion-mnist_train.csv');
test_data = readtable('fashion-mnist_test.csv');

%Split train and test data in labels and the pixels
X_train = table2array(train_data(:, [2:end])); %Pixels train set
y_train = table2array(train_data(:, 1)); %Labels

X_test = table2array(test_data(:, [2:end])); %Pixels test set
y_test = table2array(test_data(:, 1)); %Labels

%Save the initial target classes of the sets and transpose them to insert
%them as an input to the network or in the accuracy calculation
y_train_classes = y_train';
y_test_classes = y_test';

%One-hot-encoding
y_train = y_train == 0:max(y_train); %Create an One-Hot_Encoding of the target vector of the train test for multiclass classification 
y_test = y_test == 0:max(y_test); %Create an One-Hot_Encoding of the target vector of test set for multiclass classification 
%%
%Transpose the arrays to fit in the nodes
X_train  = X_train'; 
y_train = y_train';

X_test  = X_test'; 
y_test = y_test';

%Feature Selection for use in the future
[featureVectorTrain,hogVisualizationTrain] = extractHOGFeatures(X_train); 
[featureVectorTest, hogVisualizationTest] = extractHOGFeatures(X_test);
%% MLP Baseline Model
% Source: https://uk.mathworks.com/help/deeplearning/ref/patternnet.html
%Network Architecture
hiddenLayerSize = 100; 
net = patternnet(hiddenLayerSize, 'traingdx', 'crossentropy'); % patternnet for classification / traingdx combines adaptive learning rate with momentum training

%Different Architectures were tried on this network and we finally used
%hiddenLayerSize=100, based on the Ockham's Razor Rule and a traingdx train
%function. 'trainscg' and 'traingdb' have also been tested for hiddenLayerSize=10,
%but based on the Optimal Accuracy we used 'traingdx'

%Default Network Options
net.trainParam.epochs = 1000; %declare the training epochs
net.trainParam.goal = 0;
net.trainParam.lr = 0.01; %Learning Rate
net.trainParam.max_fail = 7; %EARLY STOPPING: Validation vectors are used to stop training early if the network performance 
                               %on the validation vectors fails to improve or remains the same for max_fail epochs in a row.
net.trainParam.mc = 0.9; %Momentum
net.trainParam.min_grad = 1e-5;
net.trainParam.show = 25;
net.trainParam.showCommandLine = false;
net.trainParam.showWindow = true;
net.layers{1}.transferFcn = 'logsig' %log-sigmoid transfer function
net.divideParam.trainRatio = 80/100; %divide the dataset in train and validation set, trainRatio=80/100
net.divideParam.valRatio = 20/100; % divide the dataset in train and validation set, valRatio=20/100
net.divideParam.testRatio = 0/100; % we will use the available test set in the end
net.trainParam.time = inf;

%Train the Network
[net tr Y] = train(net,X_train,y_train); % where net is the trained network
                                         %and tr is the training record
view(net)

%% Make Predictions based on the Trained Network
y_pred_train = net(X_train); % Predictions of the Trained Network. The outputs are the probabilities of each class based on the images
[prediction_baseline,K] = max(y_pred_train); % Keep the maximum probability for each class label in prediction_baseline and its index in K (predicted class labels)
K=K-1; % Restore the classes to their default values: From 0 to 9 instead of 1:10

perf_baseline_mlp = perform(net,y_train,y_pred_train); %measure the performance of the network
cross_entropy_baseline_mlp = crossentropy(y_train,y_pred_train) %calculate the cross entropy of the baseline model
accuracy_baseline_mlp = sum(K == y_train_classes)/numel(y_train_classes) %calculate the baseline model accuracy based on the prediction made
%%
disp('Performance MLP baseline: ' + string(perf_baseline_mlp)) %print the perfomance of the MLP baseline model
disp('Cross Entropy MLP baseline: ' + string(cross_entropy_baseline_mlp)) %print the cross entropy of the MLP baseline model
disp('Accuracy MLP baseline: ' + string(accuracy_baseline_mlp*100) + '%') %print the accuracy of the MLP baseline model

%% Grid Search MLP: Optimal Architecture and Hyperparameters Optimization
tic
run('mlp_hyperparameter_optimization.m')
toc

%% Perform the training of the algorithm with the Optimal Hyperparameters
tic
%Optimal Network Architecture
final_mlp_net = patternnet(hiddenLayerSize, 'traingdx', 'crossentropy'); % patternnet for classification / traingdx combines adaptive learning rate with momentum training

%Network Options after Grid Search
final_mlp_net.trainParam.epochs = 1000; %declare the maximum number of training epochs
final_mlp_net.trainParam.goal = 0;
final_mlp_net.trainParam.lr = optimal_lr_mlp; % Optimal Learning Rate after Grid Search
final_mlp_net.trainParam.max_fail = 9; %Validation vectors are used to stop training early if the network performance 
                               %on the validation vectors fails to improve or remains the same for max_fail epochs in a row.
final_mlp_net.trainParam.mc = optimal_momentum_mlp; % Optimal Momentum after Grid Search
final_mlp_net.trainParam.min_grad = 1e-5;
final_mlp_net.trainParam.show = 25;
final_mlp_net.trainParam.showCommandLine = false;
final_mlp_net.trainParam.showWindow = true;
final_mlp_net.layers{1}.transferFcn = 'logsig' %log-sigmoid transfer function
final_mlp_net.divideParam.trainRatio = 80/100; % Use 80% of the train set in the training stage of the optimal model
final_mlp_net.divideParam.valRatio = 20/100; % Use 20% of the train set as validation in the training stage of the optimal model to avoid overfitting
final_mlp_net.divideParam.testRatio = 0/100; % we will use the original test set for the predictions after training our model
final_mlp_net.trainParam.time = inf;

[final_mlp_net tr Y] = train(final_mlp_net,X_train,y_train); % where final_mlp_net is the optimal trained network and tr is the training record
view(final_mlp_net)
toc

%% Results of the trained MLP network on Train Data for comparison with the Final Model on test data
y_pred_train_optimal_mlp = final_mlp_net(X_train); %Feed the Network with the Train data
[prediction_train,I_train] = max(y_pred_train_optimal_mlp); %Take the maximum value from the prediction made for each image and assign it to the appropriate class
I_train=I_train-1; %Restore the classes to their default values from 0 to 9

perf_train = perform(final_mlp_net, y_train, y_pred_train_optimal_mlp); %calculate the performance
cross_entropy_train = crossentropy(y_train, y_pred_train_optimal_mlp); %calculate the cross-entropy
accuracy_train_optimal = sum(I_train == y_train_classes)/numel(y_train_classes); % calculate the accuracy on train data to compare it with the one on test data

disp('The accuracy for the MLP model with the train data after the Grid Search is: ' + string(accuracy_train_optimal*100) + '%')

%% Results of the trained network on the unseen Test data
y_pred_test_optimal_mlp = final_mlp_net(X_test); %Feed the Network with the Test data
[prediction,I] = max(y_pred_test_optimal_mlp); %Take the maximum value from the prediction made for each image and assign it to the appropriate class
I=I-1; %Restore the classes to their default values from 0 to 9

perf_test = perform(final_mlp_net, y_test, y_pred_test_optimal_mlp); %calculate the performance of the network on test data
cross_entropy_test = crossentropy(y_test, y_pred_test_optimal_mlp); %calculate the cross entropy of the network on test data
accuracy_test_optimal = sum(I == y_test_classes)/numel(y_test_classes); %calculate the accuracy of the network on test data

disp('The accuracy for the MLP model with the unseen test data after the Grid Search is: ' + string(accuracy_test_optimal*100) + '%')

%% Plot confusion Matrix MLP for Test set
plotconfusion(y_test,y_pred_test_optimal_mlp)

%% Print Final Values of Accuracies in different Stages MLP
disp('Accuracy MLP baseline: ' + string(accuracy_baseline_mlp*100) + '%') %Print the Baseline Model's Accuracy
disp('Optimal Accuracy MLP Grid Search: ' + string(optimal_accuracy_mlp*100) + '%') %Print the Optimal Accuracy of the MLP Grid Search
disp('Optimal Learning Rate MLP Grid Search: ' + string(optimal_lr_mlp)) %Print the Optimal Learning Rate extracted from the Grid Search
disp('Optimal Momentum MLP Grid Search: ' + string(optimal_momentum_mlp)) %Print the Optimal Momentum extracted from the Grid Search
disp('Average Accuracy MLP Grid Search: ' + string(avg_accuracy_mlp*100) +'%') %Print the Average Accuracy of the Grid Search
disp('Minimum Accuracy MLP Grid Search: ' + string(min_accuracy_mlp*100) +'%') %Print the Minimum Accuracy of the Grid Search
disp('The accuracy for the MLP model with the train data after the Grid Search is: ' + string(accuracy_train_optimal*100) + '%') %Print the optimal model's accuracy on train set
disp('The accuracy for the MLP model with the unseen test data after the Grid Search is: ' + string(accuracy_test_optimal*100) + '%') %Print the optimal model's accuracy on test set
