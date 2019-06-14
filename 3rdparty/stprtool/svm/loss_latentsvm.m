function [score,phi,loss,Data] = loss_latentsvm( i, Data, W )
% LOSS_LATENTSVM 
% 
% Synopsis:
%  [score,phi,loss,Data] = loss_latentsvm( i, Data )
%  [score,phi,loss,Data] = loss_latentsvm( i, Data, W )
%

    if nargin < 3, W = zeros( Data.nZ*Data.nDim, 1 ); end

    W = reshape( W, Data.nDim, Data.nZ );

    proj = W'*Data.X(:,i) + Data.Loss(:,i);

    [maxProj, predZ ] = max( proj );

    loss = Data.Loss( predZ, i)*Data.lossWeight(i);
    phi  = zeros(Data.nDim,Data.nZ);

    if predZ ~= Data.Z(i)
        phi(:,predZ)     = Data.X(:,i);
        phi(:,Data.Z(i)) = -Data.X(:,i); 
    end

    score = Data.lossWeight(i)*(maxProj - Data.X(:,i)'*W(:,Data.Z(i)) );
    
    phi = phi(:)*Data.lossWeight(i);

return;
