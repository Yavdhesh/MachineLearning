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
% we have to use theta with first row being set to zeros for regularization

% It feels awesome for me to understand these questions
% I will always use regularization for gradient descent as I undertsnad how to resolve 
%overfitting issues

% Always refer this for regularization
z=X*theta;

sig=1+exp(-1*z);
sig=ones(size(z))./sig;

J=-((y'*log(sig))+((ones(size(sig))-y)'*log(ones(size(sig))-sig)));
J=J/m;
thetawithoutzerotheta=theta(2:size(theta),:);
zeroRow=zeros(1,size(theta,2));
thetawithoutzerotheta=[zeroRow;thetawithoutzerotheta]
speThetasquare=thetawithoutzerotheta'*thetawithoutzerotheta;
speThetasquare=speThetasquare/2;
speThetasquare=speThetasquare/m;
speThetasquare=speThetasquare*lambda;
J=J+speThetasquare;

grad=X'*(sig-y);
grad=grad/m;
const=lambda/m;
const=const*thetawithoutzerotheta;
grad=grad+const;






% =============================================================

end
