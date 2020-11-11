function [R,subgrad] = risk_rrank_par( Data, W )
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
    

    for p = 1:numel(Data)
        if nargin < 2
            [rsk, sg] = risk_rrank_log(Data{p});
        else
            [rsk, sg] = risk_rrank_log(Data{p}, W);
        end
        if p == 1
            R       = rsk;
            subgrad = sg;
        else
            R       = R + rsk;
            subgrad = subgrad + sg;
        end
%         if isnan(R)
%             keyboard;
%         end
    end;
    
    R = R / numel( Data);
    subgrad = subgrad / numel( Data);
end
% EOF