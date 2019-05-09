function Tree = gen_stump_create(treeHeight, Min, Max, gridDens)
% GEN_STUMP_TREE
%
%  Tree = gen_stump_tree(treeHeight, Min, Max, gridDens)
%

    nDim = length(Min);
    idx  = d_samp( ones(nDim,1)/nDim, treeHeight);

    J = d_samp( ones(gridDens,1)/gridDens, treeHeight ); 

    th = Min(idx) + J(:).*(Max(idx)-Min(idx))/gridDens;
    Tree.idx = idx;
    Tree.th  = th;
end
