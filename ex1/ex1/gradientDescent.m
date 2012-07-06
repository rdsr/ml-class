function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
    m = length(y);
    J_history = zeros(num_iters, 1);

    for iter = 1:num_iters
        theta  -= alpha * computeDelta(X, y, theta);
        J_history(iter) = computeCost(X, y, theta);
    end
end

function delta = computeDelta(X, y, theta)
    m = length(y);
    delta = zeros(size(theta));

    for i = 1:m
        delta += (hypothesis(X(i, :), theta) - y(i)) * X(i, :)';
    end
    delta /= m;
end

function h = hypothesis(xi, theta)
    h = xi * theta;
end
