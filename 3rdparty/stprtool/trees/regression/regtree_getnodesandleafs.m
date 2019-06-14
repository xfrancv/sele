function Nodes = regtree_getnodesandleafs( Node)
%% REGTREE_GETNODESANDLEAFS 
%
% Nodes = regtree_getnodesandleafs( Tree )
%
%     if nargin < 2
%         num = 0;
%     end

    if isfield( Node, 'RootNode')
        Nodes = regtree_getnodesandleafs( Node.RootNode );
    else
%         Node.num = num;
        if Node.isLeaf 
            Nodes = {Node};
        else
            L = regtree_getnodesandleafs( Node.LeftNode );
            R = regtree_getnodesandleafs( Node.RightNode );

            Node.rightShift = numel( L )+1;
            Nodes = {Node,L{:},R{:}};
        end
    end

end