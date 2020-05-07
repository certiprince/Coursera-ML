function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialise useful values
m = length(y); % number of training examples

J = 0; % cost

sqErrors = (sigmoid(X) - y) .^ 2; %get squared errors

J = 1/(2*m) * sum(sqErrors); % compute cost


% =========================================================================

end
