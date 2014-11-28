% Initialisation
clear ; close all; clc
num_labels = 1;   % > 1 for multiclass problems 
lambda = 0;       % weight regularisation parameter (0 = unregularised)

% Load Training Data
fprintf('Loading features ...\n')
X = load('data\X_train.csv');
y = load('data\y_train.csv');
m = size(X, 1);
input_layer_size = n = size(X, 2);

%%%%%% CP - made artifically small initially to aid debugging %%%%%%%
hidden_layer_1_size = input_layer_size-50; 
hidden_layer_2_size = input_layer_size-20;

% Load the weights into Theta variables
fprintf('\nInitialising Neural Network Parameters ...\n')
init_Theta1 = randInitialiseWeights(input_layer_size, hidden_layer_1_size);
init_Theta2 = randInitialiseWeights(hidden_layer_1_size, hidden_layer_2_size);
init_Theta3 = randInitialiseWeights(hidden_layer_2_size, num_labels);
size(init_Theta1)
size(init_Theta2)
size(init_Theta3)
nn_params = [init_Theta1(:) ; init_Theta2(:) ; init_Theta3(:)];

% Forward propagation, back propagation, cost and gradient
[J, grad] = nnCostFunction(nn_params, input_layer_size, ...
                           hidden_layer_1_size, hidden_layer_2_size, ...
                           num_labels, X, y, lambda);

fprintf('Cost from initialised parameters: %f \n', J);

% checking against numerically calculated gradient...
% function to pass in to computeNumericalGradient as param
costFunc = @(p) nnCostFunction(p, input_layer_size, hidden_layer_1_size, ...
                               hidden_layer_2_size, num_labels, X, y, lambda);
numgrad = computeNumericalGradient(costFunc, nn_params);
printn = size(grad, 1);
fprintf('Compare last %5.0f results from numerical and analytical gradient...\n\n', printn);
for i = 1:printn,
   fprintf('%10.0f %10.6f %10.6f %10.6f\n', i, numgrad(i), grad(i), abs(numgrad(i)-grad(i)));
end
%disp([numgrad((size(numgrad,1)-printn):end) grad((size(numgrad,1)-printn):end)]);
fprintf(['\nThe above two columns you get should be very similar.\n' ...
         '(Left-Numerical Gradient, Right-Analytical Gradient)\n\n']);

% train the model
fprintf('\nTraining Neural Network... \n')
options = optimset('GradObj', 'on', 'MaxIter', 400);
	
[nn_params, J, exit_flag] = ...
	fminunc(@(t)(nnCostFunction(nn_params, input_layer_size, ...
                         hidden_layer_1_size, hidden_layer_2_size, ...
                         num_labels, X, y, lambda)), nn_params, options);
%%% e.g. from wk2
%%%fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

%%% attempt with fmincg...
% options = optimset('MaxIter', 200);

% costFunction = @(p) nnCostFunction(p, ...
                                   % input_layer_size, ...
                                   % hidden_layer_1_size, ...
                                   % hidden_layer_2_size, ...
                                   % num_labels, X, y, lambda);

% [nn_params, cost] = fmincg(costFunction, nn_params, options);

% Obtain Theta values back from nn_params
t1_rows = size(init_Theta1, 1);
t1_cols = size(init_Theta1, 2);
t1_cells = t1_rows * t1_cols;
Theta1 = reshape(nn_params(1:t1_cells), t1_rows, t1_cols);

t2_rows = size(init_Theta2, 1);
t2_cols = size(init_Theta2, 2);
t2_cells = t2_rows * t2_cols;
Theta2 = reshape(nn_params(t1_cells+1:t1_cells+t2_cells), t2_rows, t2_cols);

t3_rows = size(init_Theta3, 1);
t3_cols = size(init_Theta3, 2);
t3_cells = t3_rows * t3_cols;
Theta3 = reshape(nn_params(t1_cells+t2_cells+1:end), t3_rows, t3_cols);

% predict values from training data...
pred = predict(Theta1, Theta2, Theta3, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

fprintf('\nPaused, press enter to exit...');
pause
