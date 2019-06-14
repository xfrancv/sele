function Model = risk_latentsvm_model( W, Data )
% RISK_LATENTSVM_MODEL
%  Model = risk_latentsvm_model( W, Data )
%

    W = reshape( W, Data.nDim, Data.nZ );
    
    if Data.X0 
        Model.W   = W( 1:Data.nDim-1,:);
        Model.W0  = W( Data.nDim,:);
    else
        Model.W   = W;
        Model.W0  = zeros(1,Data.nZ);
    end
    
    Model.map = Data.yzmap;

end