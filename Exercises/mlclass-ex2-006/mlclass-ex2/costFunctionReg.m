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

% The Gradient Calculation
h = sigmoid(X*theta);
error = h-y;
grad =  (1/m).*(error'*X);

graduno = (theta*lambda)/m;
grad  = grad' + [0;graduno(2:end)];


%J = -y*log(h)
% Cost Calculation

lg= log(h) ;
Unolg = log(1-h);
f1= -y'*lg;
f2= (1-y)'*Unolg;
Jtemp=(f1-f2)/m;

J = Jtemp + sum((lambda*(theta(2:end)'.^2)))/(2*m);

% =============================================================

end
