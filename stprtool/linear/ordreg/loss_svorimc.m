function [proxyLoss, phi, loss] = loss_svorimc( i, Data, W )
% LOSS_SVORIMC
% 
% Synopsis:
%   [proxyLoss, phi, loss] = loss_svorimc( i, Data  )
%   [proxyLoss, phi, loss] = loss_svorimc( i, Data, W )
% Description:
%

    [nDims,nExamples] = size( Data.X );

    if nargin < 3, W = zeros( Data.nDims+Data.nY-1, 1 ); end

    j = Data.idx(i);
    
    V = W(1:nDims);
    B = W(nDims+1:end);

%    score = 1 - (repmat(V'*Data.X,Data.nY-1,1) - repmat( B(:), 1, nExamples)).*Data.EY;
    score = 1 - (repmat(V'*Data.X(:,j),Data.nY-1,1) - B(:)).*Data.EY(:,j);

    if issparse( Data.X)
        subgradV = sparse(nDims,1);
    else
        subgradV = zeros(nDims,1);
    end
    subgradB = zeros(Data.nY-1,1);

    loss      = 0;
    proxyLoss = 0;
    for k = 1 : Data.nY-1
         [pLoss,pred] = max([ score(k) 0]);
         proxyLoss    = proxyLoss + pLoss;
         if pred == 1
             subgradV    = subgradV - Data.X(:,j)*Data.EY(k,j);
             subgradB(k) = subgradB(k) + Data.EY(k,j);
             loss        = loss + 1;
         end
         
%         [crisk,pred] = max([score(k,:); zeros(1,nExamples)]);
%         R = R + sum(crisk);
%         idx = find( pred == 1);
%         subgradV = subgradV - Data.X(:,idx)*Data.EY(k,idx)';
%         subgradB(k) = subgradB(k) + sum( Data.EY(k,idx) );
    end

    phi = [subgradV ; subgradB];
    
%     subgrad = [subgradV; subgradB]/nExamples;
%     R = R/nExamples;

end
% EOF