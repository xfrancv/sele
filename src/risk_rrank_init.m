function Data = risk_rrank_init(X, predY, risk, nY)
%
%  Data = risk_rrank_init(X, predY, risk, nY)

    nTrn = size(X,2);
    
    Data.X  = zeros(size(X,1)*nY, nTrn );
    for i = 1 : nTrn
        xx              = zeros(size(X,1),nY);
        xx(:, predY(i)) = X(:,i);
        Data.X(:,i)     = xx(:);
    end
    Data.risk = risk(:);
end