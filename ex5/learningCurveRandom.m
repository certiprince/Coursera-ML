function [error_train, error_val] = ...
    learningCurveRandom(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).

% Number of training examples
m = size(X, 1);

% Initialise
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% Set loop
loop = 20;

% We evaluate the training error on the first i training
% examples (i.e., X(1:i, :) and y(1:i)).
%
% For the cross-validation error, we instead evaluate on
% the entire cross validation set (Xval and yval).
%
% Note: If using cost function (linearRegCostFunction)
%       to compute the training and cross validation error, should 
%       call the function with the lambda argument set to 0. 
%       Will still need to use lambda when running
%       the training to obtain the theta parameters.

for j = 1:loop
  % Compute train/cross validation errors using training examples 
  % X(1:i, :) and y(1:i), storing the result in 
  % error_train(i) and error_val(i)
  for i = 1:m
     % randomly select i rows
     tr = randperm(size(X, 1));
     tr = tr(1:i);
     
     % create matrix of randomly selected rows from X and y
     X_rand = X(tr,:);
     y_rand = y(tr,:);
     
     %learn parameters theta using randomly selected training set
     theta = trainLinearReg(X_rand,y_rand,lambda);

     % evaluate parameters theta on randomly selected training set
     [J, grad] = linearRegCostFunction(X_rand,y_rand,theta, 0);

     % accumulate errors for i training examples
     error_train(i) = error_train(i) + J;
     
     % CV set
     % randomly select i examples from CV set
     cv = randperm(size(Xval, 1));
     cv = cv(1:i);
     Xval_rand = Xval(cv,:);
     yval_rand = yval(cv,:);
     
     % evalue theta on CV set
     [J, grad_val] = linearRegCostFunction(Xval_rand,yval_rand,theta,0);
     
     
     error_val(i) = error_val(i) + J;
  end
end
%
error_train = error_train ./ 50;
error_val = error_val ./ 50;

% =========================================================================

end
