function Model = fern_train(X,Y,nTrees,treeHeight,getWeakTree, evalTree )
% FERN_TRAIN
%
% Model = fern_train( X, Y, nTrees, treeHeight, getWeakTree, evalTree )
%
    reg  = 0.0001;
    verb = 1;

    [nDim,nExamples] = size(X);
    nY               = max(Y);
        
    classCond = ones(2^treeHeight, nTrees, nY) / nY;
    Tree      = cell( nTrees, 1); 
   
    if verb, fprintf('Training FERN\n'); end
    for i = 1 : nTrees
        if verb, fprintf('.'); end
        if verb && ~mod(i,50), fprintf('%.1f%%\n', 100*i/nTrees); end
            
        Tree{i}  = getWeakTree();
        F        = evalTree( X, Tree{i} );
        
        for f = unique( F )
            for y = 1 : nY
                classCond(f,i,y) = sum( f == F & Y(:)' == y ) + reg;
            end
            classCond(f,i,:) = classCond(f,i,:) / sum( classCond(f,i,:) );
        end
    end

    Model.logClassCond = log( classCond ); 
    Model.prior        = hist( Y, nY );
    Model.prior        = Model.prior / sum( Model.prior );
    Model.Tree         = Tree;
    Model.evalTree     = evalTree;
end
