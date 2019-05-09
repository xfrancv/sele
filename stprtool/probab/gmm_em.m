function Model = gmm_em(X, nComp, covType, InitModel, varargin )
% GMM_EM Expectation-Maximization Algorithm for Gaussian mixture model.
% 
% Synopsis:
%  Model = gmm_em(X, nComp, covType )
%  Model = gmm_em(X, nComp, covType, InitModel )
%  Model = gmm_em(X, nComp, covType, InitModel, Options )
%
% Description:
%  Let p(x) be a Gaussian Mixture Model (GMM)
%   p(x,T) =  sum p(x|y,T) * p(y,T)
%          y=1:nY
%  where p(x|y), y=1:nY are Gaussian components, p(y) is a discrete 
%  probability distribution and T are parameters of the GMM 
%  (mean vectors, covariance matrices, values of p(y)). 
%
%  GMM_EM estimates parameters of the GMM using the EM algorithm
%  which finds a local optimal of the likelihood function
%    L(T) =     sum     log(  sum  p(x_i|y,T) * p(y,T) )
%          i=1:nExamples    y=1:nComp
%
%  The algorithm stops when either the Options.maxIter number
%  of iterations is reached or improvement in the log-likelihood
%  in the last consecutive steps is below Options.minImprovLogL.
%
%  The argument covType controls shape of the covariance matrix:
%    covType = 'full'      full covariance matrix (default)
%    covType = 'diag'      diagonal covarinace matrix
%    covType = 'spherical' spherical covariance matrix
%  For 1D examples sherical representation is always used.
%
%  The covariance matrix iC of the i-th component is represented as follows:
%  For covType = 'full' 
%    iC = Model.U(:,:,i)*diag(Model.D(:,i))*Model.U(:,:,i)'
%
%  For covType = 'diag' 
%    iC = diag( Model.D(:,i) )
% 
%  For covType = 'spherical' 
%    iC = Model.D(i)*eye(nDims,nDims)
%
%  To set minimal value of the variance use Options.minVar (default 1e-9).
%  This is used to avoid singular covariance matrices.
%
%  The GMM is initialized by one of the following ways:
%   1. The initial means are set to randomly selected nComp examples.
%      Options.init = 'random'   (default)
%      
%   2. The initial means are found by K-means algorithm. 
%        Options.init = 'kmeans'   (default)
%      The number of runs of k-means is given in Options.kMeansRuns 
%      (default 5) and the maximal number of iterations (per run) 
%      Options.kMeansRuns (default 100).
%
%   3. The initial estimated is taken as an input argument.
%      Model = gmm_em(X, nComp, covType, InitModel )
%
% Input:
%  X [nDims x nSamples] input samples.
%  nComp [1x1] number of components of the mixture model.
%  covType [string] shape of the covariance matrix: 'full' (default),
%       'diag','spherical'.
%  InitModel [struct] initial model.
%  Options [struct] with the following items:
%   .maxIter [1x1] maximal number of iterations (def 1000).
%   .minVar [1x1] minimal eigencalue of the cov matrix (def 1e-9)
%   .minImprovLogL [1x1] minimal improvement of the obj function (def 1e-6)
%   .init [string] initialization method: 'random', 'kmeans'
%   .kMeansRuns [1x1] number of runs of the k-means (def 5)
%   .kMeansMaxIter [1x1] max number of iterations of k-means (def 100)
%   .fullCov [1x1] If 1 then it returns full representation of the covariance 
%     matrices in varible Model.C (def 1). For high dimensional data 
%     (nDims >> 1) set it to 0 when the cov matrices are return in
%     decomposed form (see description above).
%    
% Output:
%  Model [struct] sith the following items:
%   .Mean [nDims x nComp] Mean values of the Gaussian components.
%   .Cov [nDims x nDims x nComp] Covariance matrices.
%   .Prior [1 x nComp] Weights of the components.
%   .logL [1 x nIter] evolution of the log-likelihood.
%
% Example:
%  load('riply_dataset','Trn');
%  Model = gmm_em( Trn.X, 4, 'full' );
%  
%  figure; hold on; 
%  ppatterns( Trn.X); 
%  pgmm( Model );
%  figure; 
%  plot( Model.logL)
%
% See also 
%   GMM_LOGPXY, GMM_LOGPX, GMM_LOGPOST
%

%%
if nargin < 3, covType = 'full'; end
if size(X,1) == 1, covType = 'spherical'; end

if nargin < 4, InitModel = []; end

if nargin < 5, Opt=[]; else Opt=c2s(varargin); end
if ~isfield( Opt, 'maxIter'),       Opt.maxIter       = 1000; end
if ~isfield( Opt, 'minVar'),        Opt.minVar        = 1e-9; end
if ~isfield( Opt, 'minImprovLogL'), Opt.minImprovLogL = 1e-6; end
if ~isfield( Opt, 'init'),          Opt.init          = 'random'; end
if ~isfield( Opt, 'verb'),          Opt.verb          = 1; end
if ~isfield( Opt, 'kMeansRuns'),    Opt.kMeansRuns    = 5; end
if ~isfield( Opt, 'kMeansMaxIter'), Opt.kMeansMaxIter = 100; end
if ~isfield( Opt, 'fullCov'),       Opt.fullCov       = 1; end


%% 
if nComp == 1
    Model   = gmm_ml(X, ones(1,size(X,2)), covType, Opt.minVar);
    
    logPxy  = gmm_logpxy(X,Model);
    logPx   = logsumexp(logPxy);
    Alpha   = ones(1, size(X,2));    
    logL    = sum( logPx );
else

    %%
    if ~isempty(InitModel) 
        Model = InitModel;
        nComp = length(Model.Prior);
        covType = Model.covType;
    else
        switch Opt.init,
            case 'random' 
             inx     = randperm(size(X,2));  
             centers = X(:,inx(1:nComp));

            case 'kmeans'
                KmOpt.nRuns   = Opt.kMeansRuns; 
                KmOpt.maxIter = Opt.kMeansMaxIter;
                centers       = k_means( X, nComp, KmOpt, [], Opt.verb );
        end

        [dummy,lab] = max( knnest( X, centers, [1:nComp], 1),[],1);
        Model       = gmm_ml(X, lab,covType, Opt.minVar);   
    end

    %% Main loop of the EM algorithm 
    logL = [];
    iter = 0;

    while 1    
      iter = iter + 1;

      %%%%%%%%%%%%%
      %% E-Step
      %%%%%%%%%%%%%
      logPxy     = gmm_logpxy(X,Model);
      logPx      = logsumexp(logPxy);
      logPost    = logPxy - repmat(logPx(:)',nComp,1);

      Alpha      = exp( logPost );
      logL(iter) = sum( logPx );

      %% Check stopping conditions
      if iter > 1 && logL(iter) - logL(iter-1) < Opt.minImprovLogL, break; end

      if iter >= Opt.maxIter, break; end

      % print progress
      if mod(iter,Opt.verb)==0, fprintf('iter %4d: logL=%f\n',iter, logL(iter)); end

      %%%%%%%%%%%%%
      %% M-step
      %%%%%%%%%%%%%
      Model = gmm_ml(X,Alpha,covType,Opt.minVar);   

    end 

    if Opt.verb, fprintf('iter %4d: logL=%f\n',iter, logL(iter)); end

end

Model.Alpha = Alpha;
Model.logL  = logL;


%% compute explicit covariance matrix representation
if Opt.fullCov
    nDims     = size(X,1);
    Model.Cov = zeros(nDims,nDims,nComp);
    for c = 1 : nComp
       switch covType
          case 'full'
              Model.Cov(:,:,c) = Model.U(:,:,c)*diag(Model.D(:,c))*Model.U(:,:,c)';
            
          case 'diag' 
              Model.Cov(:,:,c) = diag( Model.D(:,c) );            
            
          case 'spherical'
              Model.Cov(:,:,c) = eye( nDims,nDims) * Model.D(c);
       end
    end
end

return;


