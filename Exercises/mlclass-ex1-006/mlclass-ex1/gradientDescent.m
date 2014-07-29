function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

%h = theta(1)*X(:,1)+theta(2)*X(:,2);
%size(theta')

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    %h = theta(1)*X(:,1)+theta(2)*X(:,2);
    %theta = X(:,1)*theta(1)+X(:,2)*theta(2);
    
    h = X*theta;
    error = h-y;
    gradient =  error'*X;
    %gradient =  error'*X(:,2);
    %theta_change = alpha * (1/m)* sum(gradient);
    theta_change = gradient*alpha * (1/m);
    theta = theta - theta_change';
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
theta
end
