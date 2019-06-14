function [R,subgrad] = risk_rrank2( Data, W )
% RISK_HINGE Sum of hinge losses evaluating two-class linear classifier.
% 
% Synopsis:
%  [R,subgrad] = risk_hinge(Data)
%  [R,subgrad] = risk_hinge(Data,W)
%
% Description:
%   F(W) = 0.5*lambda*norm(W) + sum( C.* max(0,1-(X'*W).*Y) )
%
%   where lambda [1x1] is a constant, C [M x 1] are cost factors, 
%   X [N x M] features and Y [M x 1] are labels (+1/-1).
%
%  This function returns value and subgradient of risk R(W) at W.
%

    [nDim, nExamples] = size( Data.X );

    if nargin < 2, W = zeros(nDim,1); end
    
    R = 0;
    subgrad = zeros(nDim,1);
    for i = 1 : nExamples
        S       = repmat(Data.X(:,i),1,nExamples) - Data.X;
        score   = 1 + W'*S;
        idx     = find( score > 0 );
        R       = R + score(idx)*(Data.risk(i)<Data.risk(idx)) / nExamples;
        subgrad = subgrad + S(:,idx)*(Data.risk(i) < Data.risk(idx)) / nExamples;
    end
end
% EOF