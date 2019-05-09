function Data = risk_msvm_init( X, X0, Y, Loss )
% RISK_MSVM_INIT
%
% Synopsis:
%   Data = risk_msvm_init( X, X0, Y, Loss )
%

    Data.Y       = Y(:);
    Data.nLabels = max( Y );
    
    if X0 ~= 0
        Data.X = [ X ; X0*ones(1,numel(Y))];
    else
        Data.X = X;
    end
    Data.X0 = X0;
    
    [Data.nDims, Data.nExamples] = size( Data.X );    
    
    Data.correctScoreIdx = Y(:)'+[0:Data.nExamples-1]*Data.nLabels;   
    
    if nargin < 4
        % 0/1-loss by default
        Data.Loss  = ones( Data.nLabels, Data.nExamples );
        Data.Loss( Data.correctScoreIdx ) = 0;
    else
        % user specified loss        
        Data.Loss  = ones( Data.nLabels, Data.nExamples );
        for y = 1 : Data.nLabels
            idx = find( Y == y);
            Data.Loss(:,idx) = repmat( Loss(:,y), 1, length(idx));
        end
    end

end
% EOF