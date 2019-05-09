function [R,subgrad] = risk_ordreg(Data,W)
% RISK_ORDREGRESS
% 
% Synopsis:
%  [R,subgrad] = risk_ordreg(Data)
%  [R,subgrad] = risk_ordreg(Data,W)
%
% Description:
%

    if nargin < 2, W = zeros( Data.nDims + Data.nY, 1 ); end

    Alpha = [1:Data.nY]';

    V = W(1:Data.nDims);
    B = W(Data.nDims+1:end);

    score = V'*Data.X;
    score = Alpha*score + repmat(B, 1, Data.nExamples);

    score = Data.lossMatrix + score;

    linearTermOfRisk = V'*Data.subgradV0+B'*Data.subgradB0;

    [losses,predy] = max( score );

    R = sum( losses ) + linearTermOfRisk;

    subgradV = Data.subgradV0;
    subgradB = Data.subgradB0;

    for y=1:Data.nY
        idx         = find(predy == y);
        subgradV    = subgradV + sum(Data.X(:,idx),2)*Alpha(y);
        subgradB(y) = subgradB(y) + length(idx);
    end

    subgrad = [subgradV; subgradB] / Data.nExamples;
    R       = R / Data.nExamples;

return;
% EOF