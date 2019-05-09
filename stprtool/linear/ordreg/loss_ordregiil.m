function [proxyLoss, phi, loss] = loss_ordregiil( ii, Data, W )
% LOSS_ORDREGIIL
% 
% Synopsis:
%   [proxyLoss, phi, loss] = loss_ordregiil( i, Data, W )
%
%   proxyLoss [1x1] convex surrogate loss
%   phi       [Nx1] subgradient of the surrogate loss
%   loss      [Nx1]  
%
% Description:
%

    if nargin < 3, W = zeros( Data.nDims + Data.nY, 1 ); end

    i = Data.idx(ii);
    
    Alpha  = [1:Data.nY]';
    trueYl = Data.Y(1,i);
    trueYr = Data.Y(2,i);

    V = W(1:Data.nDims);
    B = W(Data.nDims+1:end);

    proj   = V'*Data.X(:,i);
    
    
    scoreL = Data.lossMatrix(1:trueYl,trueYl) + proj*([1:trueYl]'-trueYl) + B(1:trueYl) - B(trueYl); 
    [maxScoreL, predYl ] = max( scoreL );

    scoreR = Data.lossMatrix(trueYr:Data.nY,trueYr) + proj*([trueYr:Data.nY]'-trueYr) + B(trueYr:Data.nY) - B(trueYr); 
    [maxScoreR, predYr ] = max( scoreR );
    predYr = predYr + trueYr-1;
            
    proxyLoss = maxScoreL + maxScoreR;
    
    loss  = Data.lossMatrix(predYl,trueYl) + Data.lossMatrix(predYr,trueYr);
    
    phi1  = Data.X(:,i)*(predYl-trueYl + predYr - trueYr);
    phi2  = zeros(Data.nY,1);
    phi2(predYl) = phi2(predYl) + 1;
    phi2(predYr) = phi2(predYr) + 1;
    phi2(trueYl) = phi2(trueYl) - 1;
    phi2(trueYr) = phi2(trueYr) - 1;
        
    phi = [phi1; phi2];

return;
% EOF