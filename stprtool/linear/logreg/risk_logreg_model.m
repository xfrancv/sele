function Model = risk_logreg_model( Data, W )
% RISK_LOGREG_MODEL Create linear classifier.
%
% Synopsis:
%   Model = risk_logreg_model( Data, W )
%

    if Data.X0 ~= 0
        Model.W    = W(1:end-1);
        Model.W0   = Data.X0*W(end);
    else
        Model.W    = W;
        Model.W0   = 0;
    end
    
    Model.eval = @linclassif;
end