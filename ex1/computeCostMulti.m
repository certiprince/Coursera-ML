function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialise useful values
m = length(y); % number of training examples

J = 0; % cost

pred = X * theta; %get predictions
sqErrors = (pred - y) .^ 2; %get squared errors

J = 1/(2*m) * sum(sqErrors) % compute cost


% =========================================================================

end
