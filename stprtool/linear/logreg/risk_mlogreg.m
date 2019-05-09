function [R,subgrad] = risk_mlogreg( Data, W )
% RISK_MLOGREG evaluates multi-class logistic regression objective.
% 
% Synopsis:
%  [R,subgrad] = risk_mlogreg( Data )
%  [R,subgrad] = risk_mlogreg( Data, W )
%
% Description:
%
%
%
%  This function returns value and gradient of R(W) at W.
%

if nargin < 2        
    % evaluates risk ans subgradient at W = 0
    
    R        = sum(Data.weights)*log( Data.nY );
    subgrad  = zeros(Data.nDims+1, Data.nY);
   
    tmp = sum( Data.mu, 2) / Data.nY;
    for y = 1 : Data.nY
        subgrad(1:end-1,y) = tmp - Data.mu(:,y);
        subgrad(end,y)     = sum( Data.weights)/Data.nY - Data.mu0(y) ;
    end    
    subgrad = subgrad(:);
    
else
    
    W  = reshape( W, Data.nDims+1, Data.nY);
    V  = W(1:end-1,:);
    V0 = W(end,:)';
    
    proj = exp( V'* Data.X + repmat( V0, 1, Data.nExamples ));    
    R    =  Data.lambda*norm(V,'fro') + log( sum( proj))*Data.weights ...
            - sum( sum( V.*Data.mu )) - Data.mu0'*V0 ;

    subgrad  = zeros(Data.nDims+1, Data.nY);
   
    for y = 1 : Data.nY
        tmp                = (Data.weights(:)'.*proj(y,:)) ./ sum( proj );    
        subgrad(1:end-1,y) = Data.X * tmp(:) - Data.mu(:,y) + Data.lambda*V(:,y);
        subgrad(end,y)     = sum( tmp ) - Data.mu0(y) ;
    end
    
    subgrad = subgrad(:);
end
% EOF