function [predY,score] = fern_classif( X, Model )
% 
%  [predY,score] = fern_classif( X, Model )
%

    [nDim, nExamples] = size( X );

    logClassCond = fern_classcond( X, Model );
    
    score = logClassCond + repmat( log( Model.prior(:) ), 1, nExamples );
   
    [~,predY] = max( score );
end


