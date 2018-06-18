function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% we should not use 1 ,but hum prayog karenge ones(size(z)) ka
% jodte samay 1+matrix chalega parantu, vibhajit karte samay nhi chalega
% divide karte samay ./ use karna chahiye na ki / simple 

g=ones(size(z))+exp(-1*z);
g=ones(size(z))./g; % ./ is new learning

% this function 17/06/2018 by Yavdhesh Thakar



% =============================================================

end
