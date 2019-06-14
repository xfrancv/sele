function model=gmm_ml(X,Alpha,covType,minVar)
% GMM_ML Maximum Likelihood estimate of Gaussian Mixture model.
% 
% Synopsis:
%  model = gmm_ml(X,Alpha)
%  model = gmm_ml(X,labels)
%  model = gmm_ml(...,...,covType)
%  model = gmm_ml(...,...,covType,minVar)
% 
% Description:
%  Let p(x) be a Gaussian Mixture Model
%
%    p(x,T) =  sum p(x|y,T) * p(y,T)
%           y=1:nY
%
%  where p(x|y,T), y=1:nY are Gaussian components, p(y,T) is a discrete 
%  probability distribution and T are parameters of the GMM 
%  (i.e, T includes mean vectors, covariance matrices and values of p(y)). 
%
%  GMM_ML(X,Alpha) returns parameters maximizing the expected likelihood 
%  
%    L(T) =     sum       sum     Alpha(y,i) * log( p(x_i|y,T) * p(y,T) )
%          i=1:nExamples  y=1:nY
%
%  where Alpha(:,i) is a distribution which defines soft assignment of
%  i-th example X(:,i) to each of nY Gaussian components. 
%
%  GMM_ML(X,labels) is useful when the distributions Alpha are crisp,
%  i.e. max(Alpha(:,i)) = 1 for all i=1:nExamples. Then GMM_ML(X,Alpha) 
%  is the same as GMM_ML(X,labels) with labels obtained by
%  [dummy,labels] = max(Alpha). This situation occures when the GMM is 
%  to be estimated from supervised examples, i.e. each example is assigned
%  just to a single component. 
%
%  GMM_ML(X,arg1) calls GMM_ML(X,labes) if size(arg1,1) == 1 else
%  it calls GMM_ML(X,Alpha).
%
%  The shape of the covariance matrix is controled by the input 
%  argument cov_type 
%   cov_type = 'full'      full covariance matrix (default)
%   cov_type = 'diag'      diagonal covarinace matrix
%   cov_type = 'spherical' spherical covariance matrix
%
% Input:
%  X        [nDim x nExamples] Feature vectors.
%
%  Alpha    [nY x nExamples] Assignment of examples to components.
%  labels   [1 x nExamples] Labels must be from 1:nY.
% 
%  covtype [string] Type of covariacne matrix (default 'full').
%  minVar  [1x1] Definies lower bound on the minimal eigenvalue 
%           of the covariance matrices.
%
% Output:
%  model [struct] estimated Gaussian Mixture Model.
%   .Mean  [nDim x nY] Mean vectors.
%   .Prior [nY x 1] Priory probabilities p(y).
%
%  1. Full covariance matrix (cov_type='full')
%   .U   [nDim x nDim x nY] Unitary matrices
%   .D   [nDim x nY] Diagonals
%  and the covariance matrix of th y-th component is 
%     U(:,:,y)*diag(D(:,y))*U(:,:,y)'
%
%  2. Diagonal covariance matrix (cov_type='diag')
%   .D   [nDim x nY] Diagonals
%  and the covariance matrix of th y-th component is 
%     diag(D(:,y))
%
%  3. Spherical covariance matrix (cov_type='spherical')
%   .D [nY x 1] Values on the diagonal
%  and the covariance matrix of th y-th component is 
%     D(y)*eye(nDim,nDim)
% 
% Example:
%  !!! TODO: improve the example !!! 
%  load('riply_dataset','trn_X','trn_y');
%  model = gmm_ml( trn_X, trn_y );
%  figure; hold on; ppatterns(trn_X,trn_y); pgmm( model );
%
% See also 
%

[nDim,nExamples] = size(X);

if isempty(Alpha) 
    % by default a single component (Gaussian) mixture is used
    Alpha = ones(1,nExamples);
    nY = 1;
else
    % check if labels are given; if yes transform them to weights
    if size(Alpha,1) == 1
        nY = max(Alpha(:));
        lab = Alpha(:)';
        Alpha = zeros(nY,nExamples);
        Alpha(lab+[0:nExamples-1]*nY) = 1;
    else
        nY = size(Alpha,1);
    end
end
 
if nargin < 3, covType = 'full'; end

if nargin < 4,minVar = 0; end

sumAlpha = sum(Alpha,2);

%% compute mean vectors
model.Mean = zeros(nDim,nY);
for i=1:nY
    model.Mean(:,i) = X*Alpha(i,:)'/sumAlpha(i);
end

%% compute covariance matrix
switch covType
    case 'full'
        model.U = zeros(nDim,nDim,nY);
        model.D = zeros(nDim,nY);        

    case 'diag' 
        model.D = zeros(nDim,nY);
        
    case 'spherical'
        model.D = zeros(nY,1);
end


for y=1:nY    
    XC = X - repmat(model.Mean(:,y),1,nExamples);
    
    switch covType
        case 'full'
            Cov = ((repmat(Alpha(y,:),nDim,1).*XC)*XC')/sumAlpha(y);
    		[model.U(:,:,y) D] = svd(Cov);

            d            = diag(D);
        	d0           = d;	
    		d            = max(minVar,d0);
            model.D(:,y) = d;
            
        case 'diag'
            model.D(:,y) = max(minVar, XC.^2*Alpha(y,:)'/sumAlpha(y));
            
        case 'spherical'
            model.D(y)   = max(minVar,sum(XC.^2,1)*Alpha(y,:)'/(sumAlpha(y)*nDim));
    end
    
end

model.Prior   = sumAlpha'/sum(sumAlpha);
model.covType = covType;

return; 
% EOF