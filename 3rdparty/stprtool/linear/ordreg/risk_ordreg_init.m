function Data = risk_orgreg_init( X, Y, Loss )
% RISK_ORDREG_INIT
% 
% Synopsis:
%  Data = risk_orgreg_init( X, Y )
%  Data = risk_orgreg_init( X, Y, Loss )
%
% Description:
%
    [Data.nDims, Data.nExamples] = size( X );

    % use 0/1-loss by default
    if nargin < 3
        Data.nY = max( Y );
        Loss = ones(Data.nY,Data.nY) - eye(Data.nY,Data.nY);
    else
        Data.nY = size( Loss,1);
    end
    
    Data.X = X;
    Data.Y = Y;
    
    Alpha = [1 : Data.nY]';

    subgradV0 = zeros(Data.nDims,1);
    subgradB0 = zeros(Data.nY,1);
    for y = 1 : Data.nY
        idx = find( Y == y);
        subgradV0    = subgradV0 + sum(X(:,idx),2)*Alpha(y);
        subgradB0(y) = length(idx);
    end
    Data.subgradV0 = -subgradV0;
    Data.subgradB0 = -subgradB0;
   
    %
    Data.lossMatrix = zeros( Data.nY, Data.nExamples);
    for y = 1 : Data.nY
        idx = find( Data.Y == y );
        Data.lossMatrix(:,idx) = repmat(Loss(:,y),1,length(idx));
    end      
    
end
% EOF