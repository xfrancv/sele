function [score, phi, loss, Data] = loss_msvm_sparse( i, Data, W )
% LOSS_MSMV_SPARSE
%
% Synopsis:
%   [score, phi, loss, Data] = loss_msvm_sparse( i, Data )
%   [score, phi, loss, Data] = loss_msvm_sparse( i, Data, W )
%
% Description:
%   
%   

[nDim, nExamples] = size( Data.X );

if nargin < 3
    nLabels = max( Data.Y );   
    score   = 1;
    loss    = 1;
    phi     = sparse(nDim,nLabels);

    if Data.Y(i) == 1,
        phi(:,1) = -Data.X(:,i);
        phi(:,2) = Data.X(:,i);;
    else
        phi(:,1) = Data.X(:,i);
        phi(:,Data.Y(i)) = -Data.X(:,i);;
    end
    
    phi = phi(:);
    return;
end

nLabels = length(W)/nDim;
yy      = Data.Y(i);
if min(size(W)) == 1, W = reshape( W, nDim, nLabels ); end
score   = W'*Data.X(:,i) + double([1:nLabels]' ~= yy );

[maxScore,y] = max(score);
score        = maxScore - W(:,yy)'*Data.X(:,i);
loss         = double( y ~= yy );

phi       = sparse(nDim,nLabels);
phi(:,y)  = Data.X(:,i);
phi(:,yy) = phi(:,yy) - Data.X(:,i);
phi       = phi(:);

return;










