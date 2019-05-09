function model = pca(X,arg1)
% PCA Principal Component Analysis.
%
% Synopsis:
%  model = pca(X)
%  model = pca(X,outDim)
%  model = pca(X,explainedVar)
%
% Description:
%  This function computes orthonormal (i.e., model.W is orthonormal)
%  linear transform
%    y = model.W'*x + model.W0
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
%   PcaModel = pca(X,2);                       % find PCA model
%   Y = affinemap(X, PcaModel);                   % extract principal components
%   figure; ppatterns(X); axis equal;
%   figure; ppatterns(Y); axis equal;
%
% See also 
%  LINMAP, LDA
%

% get dimensions
[inDim,nVectors] = size(X);
if nargin < 2, 
    arg1 = inDim; 
end

%
model.mean = mean(X,2);

% the following code is efficient implementation of S = cov(X',1);
xc = bsxfun(@minus,X,model.mean);  % Remove mean
S = xc * xc' / nVectors;
clear xc;

% eigenvalue decomposition
[U,D,V]=svd(S);

latent = diag(D);

model.explained = cumsum(latent)/sum(latent);

% decide about the output dimension
if arg1 >= 1,
  outDim = arg1;   
else
  outDim = min(find( model.explained >= arg1));
end

% 
model.W = V(:,1:outDim);
model.W0 = -model.W'*model.mean;
model.eval = 'linmap';

return;
