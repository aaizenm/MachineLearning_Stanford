function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
    
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%c =[1:10];

X = [ones(m, 1) X];
a = sigmoid(X * Theta1');
a = [ones(size(a,1), 1) a];
h = sigmoid(a * Theta2');

thetaNoZero1 =[[zeros(size(Theta1,1),1)] Theta1(:,2:size(Theta1,2)) ]; 
thetaNoZero2 =[[zeros(size(Theta2,1),1)] Theta2(:,2:size(Theta2,2))]; 
%thetaNoZero1 = [ [ 0 ] Theta1([2:length(Theta1)]) ];
%thetaNoZero2 = [ [ 0 ] Theta2([2:length(Theta2)]) ];
%display( num2str(size(thetaNoZero1)),'size size(thetaNoZero1)' )
%display( num2str(size(thetaNoZero2)),'size size(thetaNoZero2)' )
yVector = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels)
size(yVector)
cost = -y .* log(y) - (1 - y) .* log(1 - y)

J = (1 / m) * sum(cost) + (lambda / (2 * m)) * ( sum(sum(thetaNoZero1 .^ 2))+ sum(sum(thetaNoZero2 .^ 2)));

%display( num2str(size(a)),'size a' )
%display( num2str(size(y)),'size y' )
%display( num2str(size(X')),'size X t' )
%display( num2str(size(h)),'size h' )
%display( num2str(size(h(:,1))),'size h(:,1))' )
%display( num2str(size(y(:,1))),'size y(:,1))' )

Theta1_grad = (1 / m) .* (X' * (a - y*ones(size(a,2),1)')) + sum(sum((lambda / m) .* thetaNoZero1));

Theta2_grad = (1 / m) .* (X' * (h - y*ones(size(h,2),1)')) + sum(sum((lambda / m) .* thetaNoZero2));
 

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
