function [R,subgrad] = risk_msvm( Data, W )
% RISK_MSVM
% 
% Synopsis:
%   [R,subgrad] = risk_msvm( Data, W )
%

    if nargin < 2, W = zeros( Data.nDims*Data.nLabels, 1 ); end

    W = reshape( W, Data.nDims, Data.nLabels );

    score = W'*Data.X;

    [maxScore,predY] = max( score + Data.Loss );

    R = sum( maxScore - score( Data.correctScoreIdx ) ) / Data.nExamples;

    subgrad = zeros(Data.nDims,Data.nLabels);
    for y = 1 : Data.nLabels
        idx1 = find( predY==y );
        idx2 = find( Data.Y==y );
        subgrad(:,y) =  subgrad(:,y) + sum( Data.X(:,idx1),2) - sum( Data.X(:,idx2),2 );
    end

    subgrad = subgrad(:) / Data.nExamples;

return;