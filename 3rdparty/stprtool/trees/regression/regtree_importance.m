function [importance,usage] = regtree_importance( TreeOrNode, numInVars )
%%
% importance = regtree_importance( Tree )
%
% It computes sum of relative improvements due to using individual
% input variables.
% 
    if nargin >= 2
        vec = zeros( numInVars, 1);
    end

    if isfield( TreeOrNode, 'RootNode')
        [importance,usage] = regtree_importance( TreeOrNode.RootNode, TreeOrNode.numInVars );     
        importance = importance;
    elseif ~TreeOrNode.isLeaf 
        [impLeft, usageLeft]    = regtree_importance( TreeOrNode.LeftNode, numInVars );
        [impRight,usageRight]   = regtree_importance( TreeOrNode.RightNode,  numInVars);
        
        vec( TreeOrNode.inVarIdx) = 1;

        importance = impLeft   + impRight   + vec*TreeOrNode.sseImprov;        
        usage      = usageLeft + usageRight + vec;
    else
        importance = vec;
        usage      = vec;
    end
end