function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);


%   1)    Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

a1 = [ones(m,1) X];
z2 = a1 * Theta1';
a2 = [ones(m,1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

%regularisation terms

reg1 = trace(Theta1(:,2:end)'*Theta1(:,2:end));
reg2 = trace(Theta2(:,2:end)'*Theta2(:,2:end));

J = (1/m)*trace(-y_matrix'*log(a3)-(1-y_matrix')*log(1-a3)) + (lambda/(2*m))*(reg1+reg2);

%    2)   Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients

d3 = a3 - y_matrix;
d2 = (d3 * Theta2(:,2:end)) .* sigmoidGradient(z2);
Delta1 = d2' * a1;
Delta2 = d3' * a2;
Theta1_grad = (1/m)*Delta1;
Theta2_grad = (1/m)*Delta2;


%    3)   Implement regularization with the cost function and gradients.

Theta1(:,1) = 0;
Theta2(:,1) = 0;
Theta1 = (lambda/m)*Theta1;
Theta2 = (lambda/m)*Theta2;

% Update gradients with regularisation terms
Theta1_grad = Theta1_grad + Theta1;
Theta2_grad = Theta2_grad + Theta2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
