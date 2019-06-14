function Model = loss_latentsvm_model( W, Data )
% LOSS_LATENTSVM_MODEL
%  Model = loss_latentsvm_model( W, Data )
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