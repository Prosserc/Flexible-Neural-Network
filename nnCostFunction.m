function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_1_size, ...
                                   hidden_layer_2_size, ...
                                   num_labels, ...
                                   X, y, lambda)
   %NNCOSTFUNCTION Implements the neural network cost function for a
   %neural network which performs classification
   
   % unroll nn_params
   t1_size = hidden_layer_1_size*(input_layer_size+1);
   Theta1 = reshape(nn_params(1:t1_size), hidden_layer_1_size, (input_layer_size+1));
   t2_size = hidden_layer_2_size*(hidden_layer_1_size+1);
   Theta2 = reshape(nn_params(t1_size+1:t1_size+t2_size), ...
                    hidden_layer_2_size, (hidden_layer_1_size+1));
   t3_size = num_labels*(hidden_layer_2_size+1);
   Theta3 = reshape(nn_params(t1_size+t2_size+1:end), ...
                    num_labels, (hidden_layer_2_size+1));
   
   % initialise variables
   m = size(X, 1);
   J = 0;
   Theta1_grad = zeros(size(Theta1));
   Theta2_grad = zeros(size(Theta2));
   Theta3_grad = zeros(size(Theta3));

   % forward propagation
   l1 = [ones(m, 1) X];
   z2 = l1 * Theta1';
   l2 = [ones(m, 1) sigmoid(z2)];
   z3 = l2 * Theta2';
   l3 = [ones(m, 1) sigmoid(z3)];
   z4 = l3 * Theta3';
   H = l4 = sigmoid(z4); % the hypothesis
   
   % cost for current theta params
   reg = (lambda/(2*m))*sum(sum(Theta1(:, 2:end).^2, 2))+...
                        sum(sum(Theta2(:, 2:end).^2, 2))+...
                        sum(sum(Theta3(:, 2:end).^2, 2));
   J = ((1/m)*sum(sum((-y).*log(H)-(1-y).*log(1-H), 2))) + reg;
   
   % back propagation (for gradients - derivative terms to pass into optimisation)
   % calc delta (error) in each layer of network...
   d4 = H-y;
   d3 = d4*Theta3(:, 2:end).*sigmoidGradient(z3);
   d2 = d3*Theta2(:, 2:end).*sigmoidGradient(z2);
   % calc gradients for deltas (with regularisation)...
   Theta1_grad = ((d2'*l1) ./ m)+(lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
   Theta2_grad = ((d3'*l2) ./ m)+(lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
   Theta3_grad = ((d4'*l3) ./ m)+(lambda/m)*[zeros(size(Theta3, 1), 1) Theta3(:, 2:end)];

   % unroll gradients
   grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];
   
end
