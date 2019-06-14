function y = gbrt_marginals( Gbrt, inVarIdx, x )
%%

    nT = numel( Gbrt.Tree );
    N  = size( x, 2 );
    
    cy = zeros( nT, N );
    for t = 1 : numel( Gbrt.Tree )
        cy(t,:) = regtree_marginals( Gbrt.Tree{t}, inVarIdx, x );
    end

    y = sum( cy, 1);
end