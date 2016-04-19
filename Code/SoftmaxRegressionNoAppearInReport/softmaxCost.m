function [cost, newtheta] = softmaxCost(theta, numClasses, inputSize, alpha, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% % Unroll the parameters from theta
% theta = reshape(theta, numClasses, inputSize);

numCases = length(labels);

groundTruth = full(sparse(double(labels), 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.


M = bsxfun(@minus,theta*data,max(theta*data, [], 1));
H = exp(double(M));
p = bsxfun(@rdivide, H, sum(H));
cost = -1/numCases * groundTruth(:)' * log(p(:));
thetagrad = -1/numCases * (groundTruth - p) * data';

newtheta=(theta-alpha*thetagrad)';

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
%grad = [thetagrad(:)];
end

