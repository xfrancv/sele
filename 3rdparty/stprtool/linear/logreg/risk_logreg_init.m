function Data = risk_logreg_init( X, X0, Y, lambda )
% RISK_LOGREG_INIT train two-class logistic regression classifier.
%
%   Data = risk_logreg_init( X, X0, Y, lambda )
%
% Description:
%   
%
% Input:
%   X      [nDims x nExamples] input features.
%   X0     [1x1] constant feature 
%   Y      [nExamples x 1] labels +1 or -1
%   lambda [1x1] regularization parameter (can be 0)
%
    if nargin < 4, lambda = 0; end

    Data.X      = X;
    Data.X0     = X0;
    Data.Y      = Y;
    Data.lambda = lambda;

    Data.Y(find(Y~=1)) = -1;
    
end
