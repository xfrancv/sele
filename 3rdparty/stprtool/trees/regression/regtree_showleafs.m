function regtree_showleafs( TreeOrNode )
%%
%
    if isfield( TreeOrNode, 'RootNode')
        regtree_showleafs( TreeOrNode.RootNode );
    else
        if TreeOrNode.isLeaf 
            fprintf('node=%3d, depth=%3d, trnExamples=%3d, value=%.8f, stopreason=%s\n', ...
                TreeOrNode.id, TreeOrNode.depth, TreeOrNode.trnExamples ,TreeOrNode.value, TreeOrNode.stop );
        else
            regtree_showleafs( TreeOrNode.LeftNode );
            regtree_showleafs( TreeOrNode.RightNode );
        end
    end
end