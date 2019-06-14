function Y=affinemap(X, model)
% AFFINEMAP Linear data mapping.
%
% Synopsis:
%   Y = affinemap(X, model)
%
% Description:
%  This function computes
%    Y(:,i) = model.W'*X(:,i) + model.W0     for i=1:size(X,2)
%
% Input:
%  X [inDim x nVectors] Input vectors.
%  model [struct] linear mapping:
%   .W [inDim x outDim] Projection matrix.
%   .W0 [outDim x 1] Bias.
%
% Output:
%  Y [outDim x nVectors] Output vectors.
% 
% Example:
%  help pca;
%  help lda;
%

Y = model.W'*X + repmat(model.W0(:),1,size(X,2));

return;
