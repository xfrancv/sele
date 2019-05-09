function Model = risk_msvm_model( Data, W )
% RISK_MSVM_MODEL
%
% Synopsis:
%   Model = risk_msvm_model( Data, W )
%

    W = reshape(W, Data.nDims, Data.nLabels );

    if Data.X0
        Model.W0 = Data.X0*W(end,:)';
        Model.W  = W(1:end-1,:);
    else
        Model.W0 = zeros( Data.nLabels, 1 );
        Model.W  = W;
    end
    
    Model.eval = @linclassif;

end