function [bestCenters,bestLabels,minErr] = k_means(X,nCenters,Opt,initCenters,verb)
% K-MEANS K-means clustering algorithm.
% 
% Synopsis:
%  [centers,labels,err] = k_means(X,nCenters)
%  [centers,labels,err] = k_means(X,nCenters,Opt)
%  [centers,labels,err] = k_means(X,nCenters,Opt,initCenters)
%
% Description:
%  This function implements the k-means clustering algorithm.
%   
%  If the init_centers are not given then they are selected 
%  randomly from the input vectors. 
%
% Input:
%  X [nDim x nVectors] Matrix of column vectors to be clustered.
%  nCenters [1 x 1] Number of cluster centers. 
%  Opt [struct] optimization options:
%   .maxIter [1 x 1] Maximal number of iterations (default inf).
%   .nRuns   [1 x 1] Number of runs (default 1).
%   .verb    [1 x 1] If 1 then it indicates a new iteration. Default 1.
%  
%  init_centers [nDim x nCenters] Initial cluter centers.
%  
% Output:
%  centers [nDim x Centers] Found cluster centers.
%  labels  [nVectors x 1]   Assignment to the closest center. 
%  err     [1 x 1] Sum of   Squared differences to cloasest center.
%  
% Example:
%  load('riply_dataset','Trn');
%  nCenters = 4;
%  [centers,labels,objval] = k_means( Trn.X, nCenters);
%  figure; 
%  ppatterns( Trn.X,labels); 
%  ppatterns(centers,[1:nCenters],'BigCircles');
%
%

[nDim,nVectors] = size(X);

%%
if nargin < 3, Opt = []; end
if ~isfield(Opt,'maxIter'), Opt.maxIter = inf; end
if ~isfield(Opt,'verb'),    Opt.verb    = 1; end
if ~isfield(Opt,'nRuns'),   Opt.nRuns   = 1; end

minErr = inf;
for run = 1 : Opt.nRuns

    if nargin < 4 || isempty(initCenters) || run > 1
        idx = randperm(nVectors);
        centers = X(:,idx(1:nCenters));
    else
        centers = initCenters;
    end

    old_labels = zeros(1,nVectors);
    nIter      = 0;

    %% main loop
    if Opt.verb, fprintf('Run %d: [', run); end

    while nIter < Opt.maxIter

      if Opt.verb, fprintf('.'); end
      nIter = nIter + 1;

      %% 
      out = knnest(X, centers, [1:nCenters], 1);
      [dummy, labels] = max(out);

      if all(labels == old_labels), break; end
      old_labels = labels; 

      %%
      singular_center = true( nCenters, 1);
      for i = 1 : nCenters
        idx = find(labels == i);

        if ~isempty(idx),
            singular_center(i) = false;
            centers(:,i)       = sum(X(:,idx),2)/length(idx);
        end
      end

      %% remove singular centers
      if any(singular_center)
          centers(:,singular_center) = [];
          nCenters = sum(~singular_center);
      end

    end
    
    % compute error
    out = knnest(X, centers, [1:nCenters], 1);
    [dummy, labels] = max(out);
    err = 0;
    for i = 1 : nCenters
        idx  = find(labels == i);
        N    = length( idx );
        erri = sum( sum( (X(:,idx) - repmat(centers(:,i),1,N)).^2) );
        err  = err + erri;
    end
    
    if minErr > err
        minErr       = err;
        bestCenters  = centers;
        bestLabels   = labels;
    end
    
    if Opt.verb, fprintf('] err=%f, done.\n', err); end    
end

return;
% EOF
