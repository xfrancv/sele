function Model = risk_ordreg_model( Data, W )
% RISK_ORDREG_MODEL Create linear ordinal classifier.
%
% Synopsis:
%   Model = risk_orderg_model( Data, W )
%

    V = W(1:Data.nDims);
    B = W(Data.nDims+1:end);
    Model.W = V*[1:Data.nY];
    Model.W0 = B;
    Model.eval = @linclassif;
end