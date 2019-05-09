function Tree = regtree_train( trnX, trnY, Opt )
%% REGTREE_CART train regression tree
%
%   Tree = regtree_train( trnX, trnY, Opt )
%

    if nargin < 3, Opt = []; end
    if ~isfield( Opt, 'minLeafExamples' ), Opt.minLeafExamples = 10; end
    if ~isfield( Opt, 'minMseImprov' ),    Opt.minSseImprov    = 1e-6; end
    if ~isfield( Opt, 'maxDepth' ),        Opt.maxDepth        = 100; end

    global leafNodeCnt;
    leafNodeCnt = 0;
    
    [Tree.RootNode,Tree.numNodes] = regtree_createnode( trnX, trnY, 1, Opt );
    Tree.numLeafs  = leafNodeCnt;
    Tree.numInVars = size( trnX, 1);
    
end