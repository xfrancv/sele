function Data = risk_hinge_init( X, X0, Y, C )
% RISK_HINGE_INIT Prepare data for evaluation of linear SVM risk.
% 
% Synopsis:
%  Data = risk_hinge_init( X, X0, Y )
%  Data = risk_hinge_init( X, X0, Y, C )
%

    if nargin < 4
        C = ones( length(Y), 1)/length(Y); 
    end

    if X0 ~= 0
        Data.X  = [ X; X0*ones(1, numel( Y )) ];
    else
        Data.X  = X;
    end
        
    Data.C             = C(:);
    Data.X0            = X0;
    Data.Y             = Y(:);    
    Data.Y(find(Y~=1)) = -1;

end
% EOF