function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Compute hypothesis
hyp = sigmoid(X*theta);

% Set theta0 = 0, don't include in regularisation terms (index 1)
theta(1) = 0;

% Compute cost
J = (1/m)*sum(-y'*log(hyp)-(1-y')*log(1-hyp)) + (lambda/(2*m)) * theta'*theta;

% Compute gradient
grad = (1/m)*X'*(hyp - y) + (lambda/m)*theta;

% =============================================================
grad = grad(:);
end
