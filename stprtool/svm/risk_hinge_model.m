function Model = risk_hinge_model( Data, W )
% RISK_HINGE_MODEL Create linear classifier.
%
% Synopsis:
%   Model = risk_hinge_model( Data, W )
%

    if ~iscell(Data)
	X0 = Data.X0;
    else
	X0 = Data{1}.X0;
    end

    if X0 ~= 0
        Model.W    = W(1:end-1);
        Model.W0   = X0*W(end);
    else
        Model.W    = W;
        Model.W0   = 0;
    end
    
    Model.eval = @linclassif;
end