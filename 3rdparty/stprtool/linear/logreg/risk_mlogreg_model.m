function Model = risk_logreg_model( Data, W )
% RISK_MLOGREG_MODEL Create linear classifier.
%
% Synopsis:
%   Model = risk_mlogreg_model( Data, W )
%

    if Data.X0 ~= 0
        W          = reshape( W, Data.nDims+1,Data.nY);
        Model.W    = W(1:end-1,:);
        Model.W0   = W(end,:)';
    else
        Model.W    = reshape( W, Data.nDims, nY);
        Model.W0   = zeros( nY,1);
    end
    
    Model.eval = @linclassif;
end