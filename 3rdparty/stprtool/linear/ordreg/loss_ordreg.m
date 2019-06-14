function [score, phi, loss] = loss_msvm( i, Data, W )
% LOSS_ORDREG
% 
% Synopsis:
%   [score, phi, loss] = loss_ordreg( i, Data, W )
%
% Description:
%

    if nargin < 3, W = zeros( Data.nDims + Data.nY, 1 ); end

    Alpha = [1:Data.nY]';
    truey = Data.Y(i);

    V = W(1:Data.nDims);
    B = W(Data.nDims+1:end);

    score = Alpha*(V'*Data.X(:,i)) + B + Data.lossMatrix(:,i);

    [maxScore,predy] = max( score );

    score = maxScore - V'*Data.X(:,i)*truey - B( truey );
    
    loss = Data.lossMatrix(predy,i);

    subgradV = Data.X(:,i)*(predy - truey );
    subgradB = zeros( Data.nY, 1);
    
    subgradB(predy) = 1;
    subgradB(truey) = subgradB(truey) - 1;
        
    phi = [subgradV; subgradB];

return;
% EOF