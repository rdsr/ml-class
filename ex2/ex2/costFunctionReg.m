function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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
hx = zeros(m, 1);
for i = 1:m
    hx(i) = sigmoid(theta' * X(i, :)');
    J = J + (0 - y(i)) * log(hx(i)) - (1 - y(i)) * log(1 - hx(i));
    grad = grad + (hx(i) - y(i)) * X(i, :)';
end

theta_sq = zeros(size(theta));
theta_sq(1) = theta(1);
theta_sq = theta(2:end) .* theta(2:end);

J = 1/m * (J + lambda/2 * sum(theta_sq));

grad = grad/m;
grad(2:end) = grad(2:end) + lambda/m * theta(2:end);

% =============================================================

end
