function Data = loss_orgregiil_init( X, Y, loss )
% LOSS_ORDREG_INIT
% 
% Synopsis:
%  Data = loss_orgreg_init( X, Y )
%  Data = loss_orgreg_init( X, Y, loss )
%
% Description:
%
    [Data.nDims, Data.nExamples] = size( X );

    % use 0/1-loss by default
    if nargin < 3
        Data.nY = max( Y(:) );
        loss = ones(Data.nY,Data.nY) - eye(Data.nY,Data.nY);
    else
	Data.nY = size(loss,1);
    end
    
    Data.X   = X;
    Data.Y   = Y;
    Data.idx = [1:Data.nExamples];    
    
    %
    Data.lossMatrix = loss;
    
end
% EOF