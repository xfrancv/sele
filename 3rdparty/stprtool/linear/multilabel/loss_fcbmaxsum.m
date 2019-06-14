function [score, phi, loss] = loss_fcbmaxsum( i, Data, W )
% LOSS_FCBMAXSUM hamming loss in fully connected maxsum with binary variables
%
% Synopsis:
%   [score, phi, loss] = loss_fcbmaxsum( i, Data )
%   [score, phi, loss] = loss_fcbmaxsum( i, Data, W )
%
% Description:
%   
%   
    if nargin < 3, W = zeros( Data.nDims, 1 ); end

    j = Data.idx(i);
    
    Q = reshape(W(1:Data.nFeatDims*Data.nLabels),Data.nFeatDims,Data.nLabels);
    G = W(Data.nFeatDims*Data.nLabels+1:end);
    
    projQ = Q'*Data.X(:,j);
    
    V = [projQ(:) ; G(:)];
    
    losses = sum(Data.feat(:,1:Data.nLabels) ~= repmat(Data.Y(:,j)',Data.nFeat,1),2 );
    losses = losses / Data.nLabels;
        
    score  = Data.feat * V + losses;
    [maxScore, idx] = max(score);
    
    score = maxScore - Data.Phi0(:,j)'*W;
    
    loss  = losses(idx);
    
    feat  = Data.feat(idx,:);
    z     = zeros(Data.nFeatDims,Data.nLabels);
    
    for i = 1 : Data.nLabels
        if feat(i) == 1, z(:,i) = Data.X(:,j); end
    end
    
    phi = [z(:)' feat( Data.nLabels+1:end) ]';
    phi = phi - Data.Phi0(:,j);
    
end







