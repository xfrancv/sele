function logPxy = gmm_logpxy(X,model)
% GMM_LOGPXY Computes logarithm of p(x,y) for given GMM.
% 
% Synopsis:
%  logPxy = gmm_logpxy(X,model)
%
% Description:
%  Let p(x) be a Gaussian Mixture Model
%
%   p(x) =  sum p(x|y) * p(y)
%          y=1:nY
%
%  where p(x|y), y=1:nY are Gaussian components and p(y) is a discrete 
%  probability distribution. This function computes 
% 
%     logPxy(y,i) = log( p(X(:,i)|y) * p(y) ) 
%
% Input:
%  X [ nDim x nVectors] n-dimensional real vectors.
%  model [struct] Parameters of the GMM with nY Gaussian components.
%
% Output:
%  logPxy [ nY x nVectors] logPxy(y,i) = log( p(X(:,i)|y) * p(y) ).
%

nY = length(model.Prior);
[nDim,nVectors] = size(X);

logPxy = zeros(nY,nVectors);

for y=1:nY

    XC = X - repmat(model.Mean(:,y),1,size(X,2));
    
    switch model.covType
        case 'full'            
            dist = sum( (model.U(:,:,y)'*XC).^2 ...
                        ./repmat(model.D(:,y),1,size(X,2)),1);
            log_det = sum(log(model.D(:,y)));

        case 'diag'
            dist = sum(XC.^2./repmat(model.D(:,y),1,size(X,2)),1);
            
            log_det = sum(log(model.D(:,y)));
            
        case 'spherical'
            dist = sum(XC.^2./repmat(model.D(y),size(X,1),size(X,2)),1);
            
            log_det = nDim*log(model.D(y));
    end

    logPxy(y,:) = -0.5*(dist+log_det+nDim*log(2*pi)) + log(model.Prior(y));
end

return;
% EOF