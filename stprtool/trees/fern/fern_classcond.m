function logClassCond = fern_classcond( X, Model )
% fern_classif
%  logClassCond = fern_classif( X, Model )
%
    [nDim,nExamples] = size( X );
    [lth,nTrees,nY]  = size( Model.logClassCond );
    treeHeight       = log2( lth );

    logClassCond = zeros(nY, nExamples);
    for i = 1 : nTrees
       
        F = Model.evalTree( X, Model.Tree{i} );
 
        for f = unique( F )
            idx = find( f == F );
            for y = 1 : nY
                logClassCond(y,idx) = logClassCond(y,idx) + Model.logClassCond(f,i,y);
            end
        end                
    end
end