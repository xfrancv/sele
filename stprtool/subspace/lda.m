function [model,Sw,Sb,meanOfX] = lda(X,y,outDim)
% LDA Linear Discriminant Analysis.
% 
% Synopsis:
%  model = lda(X,y)
%  model = lda(X,y,outDim)
%  [model,Sw,Sb,meanOfX] = lda(...)
%
% Description:
%  This function computes linear transform of the input labaled examples 
%  which makes them better separable. The separability is measured by 
%  the ratio of the between-class and the within-class variance. 
%  
% Input:
%  X [inDim x numExamples] Input feature vectors.
%  y [numExamples x 1] Labels of the input feature vectors. The labels
%    must be positive integers 1,2,...,maxLabel.
%  outDim [1x1] Output feature dimension (default outDim = inDim).
%
% Ouput:
%  model [struct] Linear projection:
%   .W [inDim x outDim] Projection matrix.
%   .W0 [newDim x 1] Bias.
%   .mean [inDim x 1] Mean value of data.
%
%  Sw [inDim x inDim] Within-class scatter matrix.
%  Sb [inDim x inDim] Between-class scatter matrix.
%
% Example:
%  N1=1000; N2 = 1000;
%  X = [mvnrnd([-1;1],[0.2 0;0 2],N1)' ...
%       mvnrnd([ 1;1],[0.2 0;0 2],N2)'];
%  y = [ones(N1,1); 2*ones(N2,1)];
%  
%  LdaModel = lda( X, y, 1 );
%  PcaModel = pca( X, 1 );
%  ldaX = affinemap( X, LdaModel);
%  pcaX = affinemap( X, PcaModel);
%  
%  figure('name','Original examples'); ppatterns(X,y); axis equal;
%  figure('name','1st PCA component');
%  [h,x] = hist(pcaX(find(y==1)),50); plot(x,h,'r'); hold
%  [h,x] = hist(pcaX(find(y==2)),50); plot(x,h,'b');  
%  figure('name','1st LDA component');
%  [h,x] = hist(ldaX(find(y==1)),50); plot(x,h,'r'); hold on;
%  [h,x] = hist(ldaX(find(y==2)),50); plot(x,h,'b');  
%
% See also 
%  LINMAP, PCA.
%

% process input arguments
[inDim,numExamples] = size( X );
maxLabel = max( y );

if nargin < 3, 
    outDim = inDim; 
end

% compute within-class scatter matrix
meanOfX = mean( X, 2);
Sw=zeros(inDim,inDim);
Sb=zeros(inDim,inDim);

for i = 1:maxLabel
  idx = find( y==i);
  Xi = X(:,idx);
  
  % within-class scatter 
  meanOfXi = mean(X(:,idx),2);
  xc = bsxfun(@minus,X(:,idx),meanOfXi);  % Remove mean
  Sw = Sw + xc * xc';
  
  % between-class scatter
  Sb = Sb + length(idx)*(meanOfXi-meanOfX)*(meanOfXi-meanOfX)';
end

% Compute projection matrix
%[U,D,V2]=svd( inv( Sw )*Sb ); ERROR !
%model.W = V(:,1:new_dim);
%[V,D]=eig( inv( Sw )*Sb );

[V,D] = eig( Sb, Sw );
[D,idx] = sort( diag(D), 1, 'descend');

% take outDim biggest generalized eigenvectors
model.W = V(:,idx(1:outDim));
model.W0 = -model.W'*meanOfX;
model.mean = meanOfX;
model.eval = 'linmap';


return;
% EOF