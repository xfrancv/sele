function Model = dualpca(X,outDim)
% PCA Principal Component Analysis.
%
% Synopsis:
%  Model = dualpca(X)
%  Model = dualpca(X,outDim)
%
% Description:
%  This function computes orthonormal (i.e., model.W is orthonormal)
%  linear transform
%    y = Model.W'*x + Model.W0
%
%  which makes the input column vectors X uncorrelated and 
%  centered. If the output dimension is lower than the input 
%  dimension, the transform minimizes the reconstruction error. 
%
%  model = pca(X,outDim) this calling allows to specify explicitely 
%    the output dimension outDim >= 1.
%
%  model = pca(X,explainedVar) this calling allows to specify a 
%    portion of the explained variance in data where 0 <= explainedVar < 1. 
%
% Input:
%  X [inDim x nVectors] input vectors.
%  outDim [1 x 1] Output dimension; outDim > 1 (default outDim = nDim).
%  explained_var [1x1] Portion of explained variance.
%
% Ouputs:
%  model [struct] Linear projection:
%   .W [inDim x outDim] Projection matrix.
%   .W0 [outDim x 1] Bias.
%  
%   .mean [inDim x 1] mean of the input vectors.
%   .explained [inDim x 1] percentage of the total variance explained by
%      each principal component. 
%
% Example:
%   X=mvnrnd([1;1],[0.13 0.2;0.2 0.4],1000)';  % generate 2d points
%   PcaModel = dualpca(X,2);                       % find PCA model
%   Y = affinemap(X, PcaModel);                   % extract principal components
%   figure; ppatterns(X); axis equal;
%   figure; ppatterns(Y); axis equal;
%
% See also 
%  LINMAP, LDA
%

% get dimensions
[inDim,nVectors] = size(X);

%
Model.mean = mean(X,2);

% the following code is efficient implementation of S = cov(X',1);
%Xc = bsxfun( @minus, X, Model.mean);  % Remove mean
%Kc = full( Xc'*Xc );

M  = ones(nVectors,nVectors)/nVectors;
K  = X'*X;
Kc = K - M'*K - K*M + M'*K*M;
Kc = full( Kc );

[U,L] = eig( Kc );

dL        = diag( L );
[dL,idx1] = sort( dL, 'descend' );

Model.explained = cumsum( dL(1:min(length(dL),inDim)) / sum(dL(1:min(length(dL),inDim))));

idx2     = find( dL >= 1e-6 );
idx2     = idx2(1: min(length(idx2),outDim ));
U        = U(:,idx1(idx2));
invSqrtL = diag( 1./sqrt( dL(idx2)) );



%Model.W    = Xc*U*invSqrtL;
B          = (eye(nVectors,nVectors)-M)*U*invSqrtL;
Model.W    = X*B;
Model.W0   = -Model.W'*Model.mean;
Model.eval = 'affinemap';


return;
