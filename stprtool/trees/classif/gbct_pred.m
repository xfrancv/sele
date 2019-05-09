function [predY,score]  = gbct_pred( X, Brt )
%%
%   [predY,score]  = gbct_pred( trnX, Brt )
%  
%   predY equals +1 or-1
%   score equals P(Y==+1|x)
%

    score = zeros( size(X, 2), 1);
    for t = 1 : Brt.numTrees
%         if t >= 25
%             score = score + regtree_pred( X, Brt.Tree{t}, 1);
%         else
% 
%             score = score + regtree_pred( X, Brt.Tree{t}, 0 );
%         end
          score = score + regtree_pred( X, Brt.Tree{t} );
    end
    
    predY = sign2( score );

    score = 1./(1+exp(-2*score));
    
end
