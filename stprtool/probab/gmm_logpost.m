function logPost = gmm_logpost(X,model)
% GMM_LOGPOST Computes logarithm of p(y|x) for given by GMM.
% 
% Synopsis:
%  logPost = gmm_logpost(X,model)
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
%     logPost(y,i) = log( p( y | X(:,i)) )    
%
% Input:
%  X [ nDim x nVectors] Input n-dimensional real vectors.
%  model [struct] Parameters of the GMM with nY components.
%
% Output:
%  logPost [nY x nVectors x 1] logPost(y,i) = log( p(y|X(:,i)) ).
%

logPxy = gmm_logpxy(X,model);
logPx = logsumexp(logPxy);
nY = length(model.Prior);

logPost = logPxy - repmat(logPx(:)',nY,1);


% EOF