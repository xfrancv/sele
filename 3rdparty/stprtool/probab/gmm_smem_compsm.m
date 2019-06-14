function [S,M] = gmm_smem_compsm(X,model)
% GMM_SMEM_COMPSM Computes split and merge critera used in SMEM algorithm.
%
% Synopsis:
%   [S,M] = gmm_smem_compsm(model)
% 

[nDim,nY] = size( model.Mean );

logPxy = gmm_logpxy(X,model);               % [nY x nExamples]
logPx = logsumexp(logPxy);                  % [1 x nExamples]
logPost = logPxy - repmat(logPx(:)',nY,1);  % [nY x nExamples]
logPrior = log(model.Prior(:));             % [nY x 1]

%% Compute the split criterion for each Gaussian component. 
% The higher value the more suitable for split.
S = zeros(2,nY);   
for y=1:nY
    logSumPost = logsumexp( logPost(y,:)' );
    
    vect1 = logPost(y,:) - logSumPost - logPxy(y,:) + logPrior(y);
    vect2 = exp( logPost(y,:) )/exp(logSumPost);
    
    S(1,y) = vect1*vect2';
    S(2,y) = y;
end
[dummy,idx] = sort(S(1,:),2,'descend');
S = S(:,idx);

%% Compute the merge criterion for each pair of components.
% The higher value the more suitable for merge.

M = zeros(3,nY*(nY-1)/2);
cnt=0;
for y1=1:nY
    for y2=y1+1:nY
        cnt = cnt + 1;
%        M(1,cnt) = logsumexp( logPost(y1,:) + logPost(y2,:) );
        M(1,cnt) = logsumexp( (logPost(y1,:) + logPost(y2,:))' ) ...
            - 0.5*logsumexp( 2*logPost(y1,:)') - 0.5*logsumexp( 2*logPost(y2,:)');

        M(2,cnt) = y1;
        M(3,cnt) = y2;
    end
end
[dummy,idx] = sort(M(1,:),2,'descend');
M = M(:,idx);

return;