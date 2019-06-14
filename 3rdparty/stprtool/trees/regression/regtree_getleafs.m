function Leafs = regtree_getleafs( TreeOrNode )
%%
% Leafs = regtree_getleafs( TreeOrNode )
%

    if isfield( TreeOrNode, 'RootNode')
        Leafs = regtree_getleafs( TreeOrNode.RootNode );
    else
        if TreeOrNode.isLeaf 
            Leafs = TreeOrNode;
        else
            L = regtree_getleafs( TreeOrNode.LeftNode );
            R = regtree_getleafs( TreeOrNode.RightNode );
            Leafs = [L R];
        end
    end
end