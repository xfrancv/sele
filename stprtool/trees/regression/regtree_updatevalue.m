function NewTree = regtree_updatevalue( OldTree, leafValues )
%%
%   NewTree = regtree_updatevalue( OldTree, leafValues )
%

    if isfield( OldTree, 'RootNode' )
        NewTree.RootNode  = regtree_updatevalue( OldTree.RootNode, leafValues );
        NewTree.numNodes  = OldTree.numNodes;
        NewTree.numLeafs  = OldTree.numLeafs;
        NewTree.numInVars = OldTree.numInVars;
    else
        
        if OldTree.isLeaf
            NewTree       = OldTree;
            NewTree.value = leafValues( NewTree.id );
        else
            NewTree.value       = OldTree.value; 
            NewTree.trnExamples = OldTree.trnExamples; 
            NewTree.depth       = OldTree.depth; 
            NewTree.isLeaf      = OldTree.isLeaf;
            NewTree.th          = OldTree.th; 
            NewTree.inVarIdx    = OldTree.inVarIdx; 
            NewTree.sseImprov   = OldTree.sseImprov; 
            NewTree.sse         = OldTree.sse; 
            NewTree.LeftNode    = regtree_updatevalue( OldTree.LeftNode, leafValues );
            NewTree.RightNode   = regtree_updatevalue( OldTree.RightNode, leafValues );
    end

end
