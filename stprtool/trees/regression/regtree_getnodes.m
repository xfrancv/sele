function Nodes = regtree_getnodes( Node, inVarIdx )
%% REGTREE_GETNODES returns list of nodes that using given split variable.
%
% Nodes = regtree_getnodes( Tree )
%   Returns a list of all nodes.
%
% Nodes = regtree_getnodes( Tree, inVarIdx )
%   Returns a list of nodes using input variable inVarIdx.
%
% 
    if nargin < 2
        inVarIdx = [];
    end

    if isfield( Node, 'RootNode')
        Nodes = regtree_getnodes( Node.RootNode, inVarIdx );
    else
        if Node.isLeaf 
            Nodes = [];
        else
            L = regtree_getnodes( Node.LeftNode, inVarIdx );
            R = regtree_getnodes( Node.RightNode, inVarIdx );

            if isempty( inVarIdx) | Node.inVarIdx == inVarIdx 
                Nodes = [Node L R];
            else
                Nodes = [L R];
            end
        end
    end

end