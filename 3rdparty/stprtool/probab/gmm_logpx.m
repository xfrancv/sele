function logPx = gmm_logpx(X,model)
% GMM_LOGPX Computes logarithm of p(x) for given by GMM.
% 
% Synopsis:
%  logPx = gmm_logpx(X,model)
%
% Description:
%  Let p(x) be a Gaussian Mixture Model, that is
%
%   p(x) =  sum p(x|y) p(y)
%          y=1:nY
%
%  where p(x|y), y=1:nY are Gaussians components and p(y) a discrete 
%  probability distribution. This function computes 
% 
%     logPx(i) = log( p(X(:,i)) )    i=1:size(X,2)
%
% Input:
%  X [ nDim x nVectors] Input n-dimensional real vectors.
%  model [struct] Parameters of the GMM.
%
% Output:
%  logPx [nVectors x 1] logPx(i) = log( p(X(:,i)) ).
%

logPx = logsumexp(gmm_logpxy(X,model));

% EOF