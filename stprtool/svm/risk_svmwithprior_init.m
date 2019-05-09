function Data = risk_svmwithprior_init( X, X0, Y, Wprior, posCost, negCost  )
% RISK_SVM_INIT Prepare data for evaluation of linear SVM risk.
% 
% Synopsis:
%  Data = risk_svmwithprior_init( X, X0, Y, Wprior, posCost, negCost  )
%

    Data.X  = X;
    Data.X0 = X0;
    Data.Y  = Y(:);
    Data.Y(find(Y~=1)) = -1;
    
    idx = find(Y==1);
    Data.Y(idx) = Data.Y(idx)*posCost;

    idx = find(Y~=1);
    Data.Y(idx) = Data.Y(idx)*negCost;    
    
    Data.Wprior = Wprior;
            
return;
% EOF