function [score, phi, loss] = loss_msvm( i, Data, W )
% LOSS_MSVM
%
% Synopsis:
%   [score, phi, loss] = loss_msvm( i, Data )
%   [score, phi, loss] = loss_msvm( i, Data, W )
%
% Description:
%   
%   

    if nargin < 3, W = zeros( Data.nDims*Data.nLabels, 1 ); end

    W = reshape( W, Data.nDims, Data.nLabels );

    trueY = Data.Y(i);
    proj  = W'*Data.X(:,i);

    [maxScore,predY] = max( proj + Data.Loss(:,i) );
    score = maxScore - proj( trueY );

    phi          = zeros( Data.nDims, Data.nLabels);
    phi(:,predY) = Data.X(:,i);
    phi(:,trueY) = phi(:,trueY) - Data.X(:,i);
    phi          = phi(:);
    
    loss         = Data.Loss(predY,i);
end











