function J = computeCost(X, y, theta)
    m = length(y);
    J = 0;

    for i = 1:m
        J += square(hypothesis(X(i,:), theta) - y(i));
    end
    J = J / (2 * m);
end

function h = hypothesis(xi, theta)
    h = xi * theta;
end

function s = square(x)
    s = x .* x;
end
