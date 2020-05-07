function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialise useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

% Compute the cost and gradient of regularized linear 
% regression for a particular choice of theta.
% Set J to the cost and grad to the gradient.

pred = X * theta; %get predictions
sqErrors = (pred - y) .^ 2; %get squared errors

% do not include theta0 in regularisation term (index 1)
theta(1) = 0;
J = (1/(2*m)) * sum(sqErrors) + (lambda/(2*m))*theta'*theta;

grad = (1/m)*X'*(pred - y) + (lambda/m)*theta;

% =========================================================================
% return grad as column vector
grad = grad(:);

end
