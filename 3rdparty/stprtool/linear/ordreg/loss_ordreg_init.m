function Data = loss_orgreg_init( X, Y, Loss )
% LOSS_ORDREG_INIT
% 
% Synopsis:
%  Data = loss_orgreg_init( X, Y )
%  Data = loss_orgreg_init( X, Y, Loss )
%
% Description:
%
    [Data.nDims, Data.nExamples] = size( X );

    % use 0/1-loss by default
    if nargin < 3
        Data.nY = max( Y );
        Loss = ones(Data.nY,Data.nY) - eye(Data.nY,Data.nY);
    else
        Data.nY = size( Loss, 1);
    end
    
    Data.X = X;
    Data.Y = Y;
    
    %
    Data.lossMatrix = zeros( Data.nY, Data.nExamples);
    for y = 1 : Data.nY
        idx = find( Data.Y == y );
        Data.lossMatrix(:,idx) = repmat(Loss(:,y),1,length(idx));
    end      
    
end
% EOF