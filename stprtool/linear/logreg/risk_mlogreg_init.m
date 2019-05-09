function Data = risk_mlogregw_init( X, X0, Y, lambda, weights )
% RISK_MLOGREG_INIT train multi-class logistic regression classifier.
%
%   Data = risk_mlogregw_init( X, X0, Y, lambda, weights )
%
% Description:
%   
%
% Input:
%   X       [nDims x nExamples] input features.
%   X0      [1x1] constant feature 
%   Y       [nExamples x 1] labels +1 or -1
%   lambda  [1x1] regularization parameter (can be 0)
%   weights [nExamples x 1] weights 
%
    if nargin < 4, lambda = 0; end

    Data.X         = X;
    Data.X0        = 1; % X0;
    Data.Y         = Y(:);
    Data.nY        = max( Y );
    Data.nDims     = size( X, 1);
    Data.nExamples = size( X, 2);
    Data.lambda    = lambda;
    
    if nargin < 5
        Data.weights = ones( Data.nExamples,1)/Data.nExamples; 
    else
        Data.weights   = weights(:);
    end
    
    Data.mu  = zeros( Data.nDims, Data.nY);
    Data.mu0 = zeros( Data.nY, 1);
    for y = 1 : Data.nY
        idx          = find( Y == y );
        Data.mu(:,y) = X(:,idx)*Data.weights(idx);
        Data.mu0(y)  = sum( Data.weights(idx));
    end
    
end
