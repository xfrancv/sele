function [R,subgrad] = risk_hinge( Data, W )
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

    score   = 1 - (W'*Data.X).*Data.Y';    
    idx     = find( score > 0);                
    R       = sum( score(idx).*Data.C(idx)' ) ;
    subgrad = -Data.X(:,idx)*(Data.Y(idx).*Data.C(idx)) ;
end
% EOF