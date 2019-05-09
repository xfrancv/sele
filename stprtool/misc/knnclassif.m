function Y = knnclassif( X, Model )
% KNNCLASSIF K-nearest neighbour classifier
%
% Synopsis:
%   Y = knnclassif( X, Model )
%
% Input:
%  X [D x N] test features.
%  Model.X [D x N] Class templates. 
%  Model.Y [N x 1] Class labels.
%  Model.K [1 x 1] Number of nearest neighbours.
%
% Output:
%  Y [N x 1] predicted labels.
%
% Example:
%  KNN   = load('fiveclassproblem','X','Y');
%  KNN.K = 1;
%  figure;
%  ppatterns( KNN.X, KNN.Y );
%  pclassifier( KNN, @knnclassif );
% 

[~, Y] = max( knnest( X, Model.X, Model.Y, Model.K) );

return;
