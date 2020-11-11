function [R,subgrad] = risk_rrank_log( Data, W )
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
    R1 = 0;
    proj = W'*Data.X;
    alpha = zeros( nExamples,1);
    for i = 1 : nExamples
        score    = proj - proj(i);
        %expScore = exp( score);
        %R        = R + Data.risk(i) * sum( log(1+expScore));
        R       = R + Data.risk(i) * sum( logsumexp( [zeros(nExamples,1) score(:)]'));
        %A        = Data.risk(i)*(expScore./(1+expScore));
        A       = Data.risk(i)./(1+exp(-score));
        alpha    = alpha + A(:);
        alpha(i) = alpha(i) - sum(A);
            
    end
    R = R / nExamples;
    subgrad = Data.X*alpha / nExamples;
    
end
% EOF