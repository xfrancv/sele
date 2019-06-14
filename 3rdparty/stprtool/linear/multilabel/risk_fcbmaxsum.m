function [R,subgrad] = risk_landmark( Data, W )
% RISK_LANDMARK
%
% Synopsis:
%   [R,subgrad] = risk_landmark( Data )
%   [R,subgrad] = risk_landmark( Data, W )
%
% Description:
%   
%   

    if nargin < 2, W = zeros( Data.nDims, 1 ); end
    
    W = W(:);
    nExamples = length( Data.idx );

    R = 0;
    subgrad = zeros( size(W) );
    for i = 1 : nExamples
        [score, phi, loss] = loss_fcbmaxsum( i, Data, W );
        R = R + score;
        subgrad = subgrad + phi;
    end
    
    R = R / nExamples;
    subgrad = subgrad / nExamples;
end



