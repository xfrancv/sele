function [importance, usage] = gbrt_importance( Brt )
%%
% importance = gbrt_importance( Brt )
%
% It computes sum of relative improvements due to using individual
% input variables.
% 

    numTrees  = Brt.numTrees;
    if numTrees <= 1
        importance = [];
        usage      = [];
    else
        numInVars  = Brt.Tree{2}.numInVars;
        importance = zeros( numInVars, 1);
        usage      = zeros( numInVars, 1);
        
        for t = 2 : numTrees
            [cImportance, cUsage] = regtree_importance( Brt.Tree{t} );
            importance = importance + cImportance;
            usage      = usage + cUsage ;
        end
        
        importance = importance / (numTrees-1);
        usage      = usage / (numTrees-1);
    end
    
end