function Data = risk_fcbmaxsum_init( X, Y )
% RISK_FCBMAXSUM
%
% Synopsis:
%

    Data = [];  
    
    Data.X         = X;
    Data.Y         = Y;
    Data.nFeatDims = size(X,1);
    Data.nLabels   = size(Y,1); 
    Data.nExamples = size(X,2);
    Data.idx       = [1:size(X,2)];
    
    %
    N      = 2^Data.nLabels;
    Data.nFeat = N;
    nE     = Data.nLabels*(Data.nLabels-1)/2;
    Data.E = zeros(2,nE);
    cnt    = 0;
    for t1 = 1:Data.nLabels-1
        for t2 = t1+1:Data.nLabels
            cnt = cnt + 1;
            Data.E(1,cnt) = t1;
            Data.E(2,cnt) = t2;
        end
    end

    
    Data.nDims = Data.nLabels*Data.nFeatDims + nE*4;
    
    %
    Data.feat  = zeros(N, Data.nLabels + nE*4 );
    Data.label = zeros(N, 1);
    format = ['%0' num2str(Data.nLabels) 's'];
    for i = 1 : N
        binCode = double(sprintf(format,dec2bin(i-1)) == '1');        
        Data.feat(i,1:Data.nLabels) = binCode;
        
        decCode       = binCode*2.^[Data.nLabels-1:-1:0]'+1;
        Data.label(i) = decCode;
        
        edges = zeros(4,nE);
        for e = 1 : nE
            t1   = Data.E(1,e);
            t2   = Data.E(2,e);
            b1   = binCode(t1);
            b2   = binCode(t2);
            code = b1+b2*2 + 1;
            edges(code,e) = 1;
        end
        Data.feat(i,Data.nLabels+1:end) = edges(:)';
    end

    %
    Data.Phi0 = zeros( Data.nDims, Data.nExamples);
    for i = 1 : Data.nExamples
        decCode = Data.Y(:,i)'*2.^[Data.nLabels-1:-1:0]' + 1;
        idx     = find( Data.label == decCode );
        
        feat  = Data.feat(idx,:);
        z     = zeros(Data.nFeatDims,Data.nLabels);
    
        for j = 1 : Data.nLabels
            if feat(j) == 1, z(:,j) = Data.X(:,i); end
        end

        phi = [z(:)' feat( Data.nLabels+1:end) ]';
        Data.Phi0(:,i) = phi;
    end
    
    
end
