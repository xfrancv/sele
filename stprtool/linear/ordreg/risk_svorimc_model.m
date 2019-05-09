function Model = risk_svorimc_model( Data, W )
% RISK_SVORIMC_MODEL
% 
% Synopsis:
%  Model = risk_svorimc_model( Data, W )
%
% Description:
%

    V = W(1:Data.nDims);
    B = W(Data.nDims+1:end);

    Model.W  = V*[0:Data.nY-1];
    Model.W0 = [0 -cumsum(B)'];

    Model.eval = @linclassif;
    
end
