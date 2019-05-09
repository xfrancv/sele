function Data = loss_latentsvm_init( X, Z, yzmap, X0, lossWeight )
% LOSS_LATENTSVM_INIT
% 
%  Data = loss_latentsvm_init( X, Z, yzmap, X0, lossWeight )
%

    Data.nExamples  = size( X, 2);       

    if nargin < 4, X0 = true; end

    if X0
        Data.X = [ X ; ones(1,Data.nExamples)];
    else
        Data.X = X ;
    end 

    Data.Z          = Z;
    Data.nDim       = size( Data.X, 1);
    Data.nZ         = max( Z );
    Data.nY         = numel( yzmap );
    Data.yzmap      = yzmap;
    Data.X0         = X0;

    if nargin < 5
        Data.lossWeight = ones(Data.nExamples,1);
    else
        switch length(lossWeight)
            case Data.nExamples
                Data.lossWeight = lossWeight(:);
                
            case Data.nY
                Data.lossWeight = ones(Data.nExamples, 1);
                for y = 1 : Data.nY
                    idx = find( ismember( Data.Z, yzmap{ y } ) );
                    Data.lossWeight(idx) = lossWeight(y);
                end
                                
            otherwise 
                error('lossWeight has incorrect size.');
        end
    end
    
    %
    z2y = zeros( Data.nZ, 1);
    for y = 1 : Data.nY, z2y( yzmap{y} ) = y; end

    %
    Data.Loss = ones( Data.nZ, Data.nExamples );
    for z = 1 : Data.nZ
        idx = find( Z == z );
        Data.Loss( z, idx) = 0;

        idx = find( Data.Z(:) ~= z & ismember( Data.Z(:), yzmap{ z2y(z) } ) );
        Data.Loss( z, idx ) = -inf;
    end
    
return;
