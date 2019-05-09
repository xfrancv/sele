function [R,subgrad] = risk_svorimc( Data, W )
% RISK_EBC
% 
% Synopsis:
%  [R,subgrad,data] = risk_svorimc(Data)
%  [R,subgrad,data] = risk_svorimc(Data,W)
%
% Description:
%

    [nDims,nExamples] = size( Data.X );

    if nargin < 2, W = zeros( Data.nDims+Data.nY-1, 1 ); end

    V = W(1:nDims);
    B = W(nDims+1:end);

    score = 1 - (repmat(V'*Data.X,Data.nY-1,1) - repmat( B(:), 1, nExamples)).*Data.EY;

    subgradV = zeros(nDims,1);
    subgradB = zeros(Data.nY-1,1);

    R = 0;
    for k = 1 : Data.nY-1
        [crisk,pred] = max([score(k,:);zeros(1,nExamples)]);
        R = R + sum(crisk);
        idx = find( pred == 1);
        subgradV = subgradV - Data.X(:,idx)*Data.EY(k,idx)';
        subgradB(k) = subgradB(k) + sum( Data.EY(k,idx) );
    end

    subgrad = [subgradV; subgradB]/nExamples;
    R = R/nExamples;

end
% EOF