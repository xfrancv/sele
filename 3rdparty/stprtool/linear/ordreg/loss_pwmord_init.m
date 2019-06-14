function Data = loss_pwmord_init( X, Y, cutLabels )
% LOSS_PWMORD_INIT
%
% Synopsis:
%   Data = loss_pwmord_init( X, Y, cutLabels )
%

    Data.X       = X;
    Data.nDims     = size( X, 1);

    Data.Y       = Y(:);
    Data.nY      = max( cutLabels );
    
    Data.idx     = [1 : size(X,2)];
    
    Data.cutLabels = cutLabels;
    Data.nZ        = length( Data.cutLabels );
    Data.A         = zeros( Data.nZ,Data.nY);

    for z = 1 : length( Data.cutLabels )-1
        
        y1 = Data.cutLabels(z);
        y2 = Data.cutLabels(z+1);
        N  = (y2-y1)+1;
        for i = 1 : N
            alpha              = (N-i)/(N-1);
            Data.A(z,y1+i-1)   = alpha;
            Data.A(z+1,y1+i-1) = 1 - alpha;
        end
    end
    
end
% EOF