function [Model,Stat] = train_sele_linear( X, predY, predLoss, lambda, nBatches, Opt )
% 
%  Model = train_sele_linear( X, predY, predLoss, lambda, nBatches )
% 
%  [Model,Stat] = train_sele_linear( X, predY, predLoss, lambda, nBatches, Opt )
%

    if nargin < 5
        error('At least 5 input arguments must be supplied.');
    end

    if nargin < 6
        Opt.verb    = 1;   
        Opt.tolRel  = 0.01;
    end
    
    [nDims,nExamples] = size( X);
    nY                = max( predY );
    
    
    % split examples randomly to nBatches batches
    Data   = [];
    idx    = randperm(nExamples);
    from   = 1;
    for p = 1 : nBatches
        to      = round( p*nExamples/nBatches );   
        Data{p} = risk_rrank_init( [X(:,idx(from:to)) ; ones(1,to-from+1)], ...
            predY(idx(from:to)), predLoss(idx(from:to)), nY);
        from    = to + 1;
    end

    % call BMRM solver to minimize the convex risk
    [W, Stat] = bmrm( Data, @risk_rrank_par, lambda, Opt );

    
    % get the parameters
    W          = reshape(W, nDims+1, nY );
    Model.W    = W(1:nDims, :);
    Model.W0   = W(nDims+1,:)';
    Model.type ='linear';
    
end
