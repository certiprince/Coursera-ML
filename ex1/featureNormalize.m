function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% Initialise useful variables
m = size(X, 1);
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% Store mean values in mu matrix

mu = mean(X);
mu_matrix = ones(m, 1) * mu;

% Compute the standard deviation of each feature, storing
% the standard deviation in sigma matrix

sigma = std(X);
sigma_matrix = ones(m, 1) * sigma;

% Subtract mean and divide by std

X_norm = (X - mu_matrix)./sigma_matrix;

% ============================================================

end
