function Model = gbrt_train( trnX, trnY, Opt, valX, valY )
% GBRT_TRAIN Gradient boosted regression tree.
%
%  Model = gbrt_train( trnX, trnY, Opt )
%  Model = gbrt_train( trnX, trnY, Opt, valX, valY )
%

    if nargin < 3, Opt = []; end
    if ~isfield( Opt, 'verb'),            Opt.verb            = 1; end
    if ~isfield( Opt, 'numTrees'),        Opt.numTrees        = 10; end
    if ~isfield( Opt, 'minLeafExamples'), Opt.minLeafExamples = 5; end
    if ~isfield( Opt, 'minSseImprov'),    Opt.minSseImprov    = 1e-6; end
    if ~isfield( Opt, 'maxDepth'),        Opt.maxDepth        = 4; end
    if ~isfield( Opt, 'learningRate'),    Opt.learningRate    = 0.1; end
    
    if nargin < 5, compValError = 0; else compValError = 1; end

    %% 
    Model.Tree = cell( Opt.numTrees, 1);

    %% create initial tree which predicting median of outputs regardless the input 
    Tree.numNodes        = 1;
    Tree.numLeafs        = 1;
    Tree.RootNode.isLeaf = 1;
    Tree.RootNode.value  = median( trnY );
    Tree.RootNode.id     = 1;
    Tree.RootNode.trnExamples = length( trnY );
    Model.Tree{1}          = Tree;

    %%
    predY    = regtree_pred( trnX, Tree );
    difY     = trnY(:) - predY(:);

    trnMae   = mean( abs(difY));
    
    %% compute valudation error if requested
    if compValError
        valPredY = regtree_pred( valX, Tree );
        valMae   = mean(abs(valPredY(:)-valY(:)));
        
        if Opt.verb
            fprintf('t=%3d: TrnMSE = %.6f, TrnMAE = %.6f, TstMAE=%.6f\n', ...
                     0, mean( difY.^2 ), trnMae(1), valMae(1));
        end
    elseif Opt.verb
        fprintf('t=%3d: TrnMSE = %.6f, TrnMAE = %.6f\n', 0, mean( difY.^2 ), trnMae(1) );
    end

    %%
    for t = 2 : Opt.numTrees

        %% fit regression tree to sign of differences
        sgnDifY = sign2( difY );
        Tree    = regtree_train( trnX, sgnDifY , Opt );

        %% compute (median) optimal otput values
        [tmpPredY, nodeId] = regtree_pred( trnX, Tree );
        tmpDifY            = tmpPredY(:)-sgnDifY ;

        newValue = zeros( Tree.numLeafs, 1);
        for i = 1 : Tree.numLeafs
           idx         = find( i == nodeId );
           newValue(i) = median( difY(idx ) );
        end

        %% diminish newValues by learning rate to avoid overfitting
        newValue    = newValue * Opt.learningRate;

        Model.Tree{t} = regtree_updatevalue( Tree, newValue );    
        
        %% update prediction of the ensemble
        predY       = predY + regtree_pred( trnX, Model.Tree{t} );
        difY        = trnY(:) - predY(:);    

        trnMae(t)   = mean( abs(difY));
        
        
        %% compute valudation error if requested
        if compValError        
            valPredY    = valPredY + regtree_pred( valX, Model.Tree{t} );
            valMae(t)   = mean(abs(valPredY(:)-valY(:)));
            
            if Opt.verb
                fprintf('t=%3d: TrnMSE = %.6f, TrnMAE = %.6f, TstMAE = %.6f\n', ...
                    t, mean( difY.^2 ), trnMae(t), valMae(t));
            end
        elseif Opt.verb
            fprintf('t=%3d: TrnMSE = %.6f, TrnMAE = %.6f\n', ...
                t, mean( difY.^2 ), trnMae(t) );
        end
    end
    
    Model.numTrees = numel( Model.Tree );
    Model.Options  = Opt;
    Model.trnMae   = trnMae;
    
    if compValError, Model.valMae = valMae; end
    
end