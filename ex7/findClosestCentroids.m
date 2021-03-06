function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% Set m
m = size(X,1);

% You need to return the following variables correctly.
idx = zeros(m, 1);

D = zeros(m,K);

% Go over every example, find its closest centroid, and store
% the index inside idx at the appropriate location.
% Concretely, idx(i) should contain the index of the centroid
% closest to example i. Hence, it should be a value in the 
% range 1..K

% loop over all examples
for i = 1:K
    dist = bsxfun(@minus, X, centroids(i,:));
    D(:,i) = sum(dist.^2,2);
end

[M, idx] = min(D,[],2);

% =============================================================

end
