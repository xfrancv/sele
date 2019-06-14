function Data = risk_latentsvm_init( X, Z, yzmap, X0, gamma )
% RISK_LATENTSVM_INIT
% 
%  Data = risk_latentsvm_init( X, Z, yzmap, X0, gamma )
%

    if X0, Data.X = [X; ones(1,size(X,2))]; else Data.X = X; end
    
    Data.X0    = X0;
    Data.Z     = Z;
    Data.yzmap = yzmap;

    Data.nDim      = size( Data.X, 1);
    Data.nZ        = max( Data.Z );
    Data.nY        = length( yzmap );
    Data.nExamples = size(Data.X,2);
       
    if nargin < 5, Data.gamma = ones( Data.nExamples,1 ); end
    Data.gamma = Data.gamma(:);
    
    Data.Loss = ones( Data.nZ, Data.nExamples );
    
    z2y = zeros( Data.nZ, 1);
    for y = 1 : Data.nY, z2y( yzmap{y} ) = y; end
    
    for z = 1 : Data.nZ
        idx = find( Data.Z == z );
        Data.Loss( z, idx) = 0;
        
        idx = find( Data.Z(:) ~= z & ismember( Data.Z(:), yzmap{ z2y(z) } ) );
        Data.Loss( z, idx ) = -inf;
    end
    
    Data.Psi0 = zeros( Data.nDim, Data.nZ );
    for z = 1 : Data.nZ
        idx = find( Data.Z == z );
        Data.Psi0(:,z) = Data.X(:,idx)*Data.gamma(idx);
    end
end

