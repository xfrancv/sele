function Model = loss_pwmord_model( Data, W )
% LOSS_PWMORD_MODEL 
%
% Synopsis:
%   Model = loss_pwmord_model( Data, W )
%

    
    Model.W0   = W( Data.nDims*Data.nZ+1:end);
    Model.V    = reshape( W(1:Data.nDims*Data.nZ),Data.nDims,Data.nZ );
    Model.A    = reshape( Data.A,Data.nZ,Data.nY);
    Model.W    = Model.V*Model.A;
    Model.eval = @linclassif;
    
end