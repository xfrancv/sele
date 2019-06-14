function predY  = gbrt_pred( X, Brt )
%%
%   predY  = gbrt_pred( trnX, Brt )
%

    predY = zeros( size(X, 2), 1);
    for t = 1 : Brt.numTrees
        predY = predY + regtree_pred( X, Brt.Tree{t} );
    end

end
