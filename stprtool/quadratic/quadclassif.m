function [Y,score] = quadclassif( X, Model )
% QUADCLASSIF Quadratic classifier.
%
% Synopsis:
%   [Y,score] = quadclassif( X, Model )
%
% Description:
%   Two-class classifier deciding based on quadratic score function
%     f(x) = x'*H*x + x'f + W0 = qmap(x)'*W + W0,
%   where H is a symmetric matrix. 
% 
% Example:
%   TBD!
%

score = Model.W(:)'*qmap(X) + Model.W0;

Y = sign(score); Y(find(Y==0)) = 1;

end
