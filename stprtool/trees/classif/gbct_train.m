function Model = gbct_train( trnX, trnY, Opt, valX, valY )
% GBCT_TRAIN Gradient boosted classification tree
%
%  Model = gbct_train( trnX, trnY, Opt )
%  Model = gbct_train( trnX, trnY, Opt, valX, valY )
%

    if nargin < 3, Opt = []; end
    if ~isfield( Opt, 'verb'),            Opt.verb            = 1; end
    if ~isfield( Opt, 'numTrees'),        Opt.numTrees        = 10; end
    if ~isfield( Opt, 'minLeafExamples'), Opt.minLeafExamples = 5; end
    if ~isfield( Opt, 'minObjImprov'),    Opt.minSseImprov    = 1e-6; end
    if ~isfield( Opt, 'maxDepth'),        Opt.maxDepth        = 4; end
    if ~isfield( Opt, 'learningRate'),    Opt.learningRate    = 0.1; end
    
    if nargin < 5, compValError = 0; else compValError = 1; end

    %% 
    Model.Tree = cell( Opt.numTrees, 1);

    %% create initial tree which predicting median of outputs regardless the input 
    Tree.numNodes        = 1;
    Tree.numLeafs        = 1;
    Tree.RootNode.isLeaf = 1;
    Tree.RootNode.value  = 0.5*log(sum(trnY==+1)/sum(trnY==-1));
    Tree.RootNode.id     = 1;
    Tree.RootNode.trnExamples = length( trnY );
    Model.Tree{1}        = Tree;

    %%
    scoreY   = regtree_pred( trnX, Tree );
    predY    = sign2( scoreY );

    trnErr   = mean( trnY(:) ~= predY(:) );
    
    %% compute valudation error if requested
    if compValError
        valScoreY = regtree_pred( valX, Tree );
        valPredY  = sign2( valScoreY );
        valErr    = mean( valPredY(:) ~= valY(:) );
        
        if Opt.verb
            fprintf('t=%3d: TrnErr = %.4f, ValErr = %.4f\n', 1, trnErr, valErr );
        end
    elseif Opt.verb
        fprintf('t=%3d: TrnErr = %.4f\n', 1, trnErr );
    end

    %%
    for t = 2 : Opt.numTrees

        %% L2-approximation of gradient by regression tree 
        tmpTrnY = 2*trnY(:) ./ (1+exp(2*trnY(:).*scoreY(:)));
        Tree    = regtree_train( trnX, tmpTrnY , Opt );

        %% 
        [tmpPredY, nodeId] = regtree_pred( trnX, Tree );
%        tmpDifY            = tmpPredY(:)-sgnDifY ;

        newValue = zeros( Tree.numLeafs, 1);
        for i = 1 : Tree.numLeafs
           idx         = find( i == nodeId );
           newValue(i) = sum( tmpTrnY(idx)) / (eps+sum( abs(tmpTrnY(idx)).*(2-abs( tmpTrnY(idx)))));
        end

        %% diminish newValues by learning rate to avoid overfitting
        newValue    = newValue * Opt.learningRate;

        Model.Tree{t} = regtree_updatevalue( Tree, newValue );    
        
        %% update prediction of the ensemble
        scoreY      = scoreY + regtree_pred( trnX, Model.Tree{t} );
        predY       = sign2( scoreY );

        trnErr(t)   = mean( predY(:) ~= trnY(:) );
        
        
        %% compute valudation error if requested
        if compValError        
            valScoreY   = valScoreY + regtree_pred( valX, Model.Tree{t} );
            valPredY    = sign2( valScoreY );
            valErr(t)   = mean( valPredY(:)~= valY(:) );
            
            if Opt.verb
                fprintf('t=%3d: TrnErr = %.4f, ValErr = %.4f\n', t, trnErr(t), valErr(t) );
            end
        elseif Opt.verb
            fprintf('t=%3d: TrnErr = %.4f\n', t, trnErr(t) );
        end
    end
    
    Model.numTrees = numel( Model.Tree );
    Model.Options  = Opt;
    Model.trnErr   = trnErr;
    
    if compValError, Model.valErr = valErr; end
    
end
