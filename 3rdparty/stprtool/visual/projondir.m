function [Z, P]  = projondir( X, W, outDim)
% PROJONDIR projects data on a direction and PCA components.
%
% Synopsis:
%   [Z,P]  = projondir( X, w)
%   [Z,P]  = projondir( X, w, outDim)
% 
% Description:
%  This function returns a low-dimensional projection of input data, i.e.
%     Z = P*X 
%  where P(1,:) = w and P(2:outDim,:) is the are the principal components
%  of the input data X in a linear subspace orthogonal to w. This function
%  is useful for visulaization of a high-dimensional linear classifier.
%
% Input:
%  X [D x M] input data.
%  w [D x 1] first projection direction.
%  outDim [1x1] number of output dimensions (default 2).
% 
% Output:
%  Z [outDim x M] projected data.
%  P [outDim x D] projection matrix.
%
% Example:
%  load( 'riply_dataset', 'Trn' );
%  Trn.X = qmap( Trn.X );
%  [w,w0,stat] = svmocas( Trn.X,1,Trn.Y, 1);
%  
%  figure;
%  ppatterns( projondir( Trn.X, w, 2 ), Trn.Y );
%  hold on;
%  a=axis;
%  plot([-w0 -w0],[a(3:4)],'--k');
%
%  figure;
%  ppatterns( projondir( Trn.X, w, 3 ), Trn.Y );
%  hold on; 
%  grid on;
%  phyperplane([1 0 0], w0);
%

    if nargin < 3, outDim = 2; end

    proj1    = W'*X / (W'*W);

    Xort     = X - W*proj1;
    PcaModel = pca( Xort, outDim-1 );

    v1 = W;
    v2 = PcaModel.W;

    P = [v1 v2]';
    
    Z = P*X;
end
