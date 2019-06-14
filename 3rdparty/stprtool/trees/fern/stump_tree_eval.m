function F = stump_tree_eval( X, Tree )
% STUMP_TREE_EVAL
%
%   F = stump_tree_eval( X, Tree )
%

    nExamples  = size(X,2);
    treeHeight = length( Tree.idx );

    f = X(Tree.idx,: ) >= repmat( Tree.th(:), 1, nExamples );

    F = 2.^[0:treeHeight-1] * f + 1;

end
