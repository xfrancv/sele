function model = gmm_smem(X, nComp, covType, initModel, varargin )
% GMM_SMEM Split and Merge EM Algorithm estimating Gaussian Mixture Model.
%
% Synopsis:
%  model = gmm_smem(X, nComp, covType)
%  model = gmm_smem(X, nComp, covType, initModel)
%  model = gmm_smem(X, nComp, covType, initModel, options )
%  model = gmm_smem(X, nComp, covType, initModel, 'opt1',val1,...)
%
% Description:
%  Implementation of SMEM algorithm fitting parameters of GMM from data:
%    Ueda et al. Split and Merge EM algorithm for Improving Gaussian
%     Mixture Density Estimates. Neural Computation. Journal of VLSI Signal
%     Processing 26, 133-140, 2000.
%
%  SMEM algorithm solves the same task as the classical EM algorithm. 
%  See HELP GMM_EM for the task formulation. SMEM differs from the EM by
%  using "split and merge" heuristics which help to avoid local optima 
%  of the likelihood function.
%
% Input:
%   X [nDim x nSamples] Input vectors stored as columns.
%   nComp [1x1] Number of component of the GMM.
%   covType [string] Type (shape) of the covariance matrices.
%     Possible values are 'full', 'diag', 'spherical'.
%   initModel [GMM struct] Initial GMM estimate. By default initModel=[]
%     which tells the algorithm to use options.init method for
%     initialization. See HELP GMM_CREATE.
%
% options
%  .minVar [1x1] Minimal variance of the covariance matrix (default 1e-9).
%     See HELP GMM_EM. 
%  .maxIter [1x1] Maximal number of iteration of the EM algorithm
%     whenever called by SMEM (default 1000).
%  .minImprovLogL [1x1] Minimal improvement in logL in order to continue.
%     This value is used in EM, SMEM and in partial EM steps of SMEM.
%     (default 1e-9).
%  .maxCand [1x1] Maximal number of split-merge candidates to
%     try in each SMEM iteration (default 5).
%  .maxIterOfSMEM [1x1] Maximal number of iterations of the SMEM algorithm
%     (default 100).
%  .maxPartialIter [1x1] Maximal number of iterations of partial EM
%     algorithm (default 1000);
%  .init [string] Method used to setup inital model (default 'random').
%     See HELP GMM_EM. 
%  .verb [1x1] Verbosity (default 1).
%   
%
% Output:
%  model [GMM struct] Estimated GMM. See HELP GMM_CREATE.
%  
% Example:
%  load('riply_dataset','Trn');
%  Model = gmm_smem( Trn.X, 4, 'full' );
%  figure; hold on; ppatterns(Trn.X); pgmm( Model );
%   
% See also GMM_EM
%

%% 
if nargin < 3, covType = 'full'; end
if size(X,1) == 1, covType = 'spherical'; end

if nargin < 4, initModel = []; end

if nargin < 5, options=[]; else options=c2s(varargin); end
if ~isfield( options, 'maxIter'), options.maxIter = 1000; end
if ~isfield( options, 'minVar'), options.minVar = 1e-6; end
if ~isfield( options, 'minImprovLogL'), options.minImprovLogL = 1e-9; end
if ~isfield( options, 'init'), options.init = 'random'; end
if ~isfield( options, 'verb'), options.verb = 1; end
if ~isfield( options, 'maxCand'), options.maxCand = 5; end
if ~isfield( options, 'maxIterOfSMEM'), options.maxIterOfSMEM = 100; end
if ~isfield( options, 'maxPartialIter'), options.maxPartialIter = 1000; end

%% Initial run of ordinary EM 
if options.verb
    fprintf('[ Initial run of EM algorithm ]\n');
end

EMOpt = options; EMOpt.verb = 0;
model = gmm_em(X, nComp, covType, initModel, EMOpt );

if options.verb
    fprintf(' nIter=%d, LogL: init=%f, final=%f, improv=%f\n',...
        length(model.logL),model.logL(1),model.logL(end),model.logL(end)-model.logL(1));
end

logL = model.logL;

%% Main loop of the SMEM algorithm 
iter = 0;
while 1    
  iter = iter + 1;
  
  if options.verb
      fprintf('\n[ SMEM iteration %d ]\n',iter);
  end

  %% Compute split and merge criteria
  [S,M] = gmm_smem_compsm(X,model);
  
  %% Sort candidates for split and merge
  cnt = 0;
  candidates = [];
  for m=1:size(M,2)
      for s=1:size(S,2)
          if M(2,m) ~= S(2,s) && M(3,m) ~= S(2,s)
              cnt = cnt + 1;
              candidates(1:2,cnt) = M(2:3,m);
              candidates(3,cnt) = S(2,s);
          end
          if cnt >= options.maxCand, break; end
      end
      if cnt >= options.maxCand, break; end
  end
    
  %% Try candidates
  maxLogL = -inf;
  for c=1:size(candidates,2)
      
      %% Partial EM steps
      if options.verb
        fprintf(' Merge (%d,%d) and split %d\n', candidates(:,c));
        fprintf('  Partial EM:');
      end
      cand_model = partial_EM(X,model,options,candidates(:,c));
      
      if options.verb
        fprintf(' nIter=%d, LogL: init=%f, final=%f, improv=%f\n',...
            length(cand_model.logL),model.logL(end),cand_model.logL(end),cand_model.logL(end)-model.logL(end));
      end
      
        
      %% finish tuning by applying full EM steps
      if options.verb
          fprintf('  Full EM:');
      end      
      initLogL = cand_model.logL(end);
      cand_model = gmm_em(X, nComp, covType, cand_model, EMOpt );
      if options.verb
        fprintf(' nIter=%d, LogL: init=%f, final=%f, improv=%f\n',...
            length(cand_model.logL),initLogL,cand_model.logL(end),cand_model.logL(end)-initLogL);
      end
      
      if maxLogL < cand_model.logL(end)
          maxLogL = cand_model.logL(end);
          best_cand_model = cand_model;
      end
  end
  
  improv = maxLogL - model.logL(end);
  if options.verb
      fprintf('Candidates improved logL by %f \n',improv);
  end
  
  if improv < options.minImprovLogL
      break;
  end
  
  %% use the best candidate model
  logL = [logL best_cand_model.logL];
  model = best_cand_model;
  model.logL = logL;
  model.nIter = length(model.logL) + length(best_cand_model.logL);
          
  if iter >= options.maxIterOfSMEM
      break;
  end
end 

return;

%% Implementation of the split and merge rules and partial EM update
% 1. two components (m1,m2) are merged and one component (s) is split
% 2. applies EM rules to update only the three new components
% 3. the new components recplace the old one
function cand_model = partial_EM(X,cand_model,options,candidates)

m1= candidates(1);  % index of the first component to merge
m2= candidates(2);  % index of the second component to merge
s= candidates(3);   % index of the component to split

%% create partial GMM model which contains only three componets
% created by merging old components m1 and m2 to one new component and 
% split old component s to two new componets
nDim = size(X,1);
model.Mean = zeros(nDim,3);

% merge means of m1 and m2 to single one
alpha1 = cand_model.Prior(m1)/(cand_model.Prior(m1)+cand_model.Prior(m2));
alpha2 = cand_model.Prior(m2)/(cand_model.Prior(m1)+cand_model.Prior(m2));
model.Mean(:,1) = alpha1*cand_model.Mean(:,m1)+alpha2*cand_model.Mean(:,m2);

% split mean s to two new components
%noise_magn = norm(cand_model.Mean(:,s))*1e-1;
noise_magn = norm(cand_model.Mean(:,s))*0.1/sqrt(nDim);
model.Mean(:,2) = cand_model.Mean(:,s)+randn(nDim,1)*noise_magn;
model.Mean(:,3) = cand_model.Mean(:,s)+randn(nDim,1)*noise_magn;

% update priors
model.Prior = [cand_model.Prior(m1)+cand_model.Prior(m2) ...
               cand_model.Prior(s)*0.5 ...
               cand_model.Prior(s)*0.5];

% Split and merge covariance matrices. This step depends on represenation
% of the covariance matrices
covType = cand_model.covType;
model.covType = covType;
switch covType
    case 'full'
        model.U = zeros(nDim,nDim,3);
        model.D = zeros(nDim,3);
        
        C1 = cand_model.U(:,:,m1)'*diag(cand_model.D(:,m1))*cand_model.U(:,:,m1);
        C2 = cand_model.U(:,:,m2)'*diag(cand_model.D(:,m2))*cand_model.U(:,:,m2);
        C = alpha1*C1 + alpha2*C2;
        [U S V] = svd(C);
        model.U(:,:,1) = U;
        model.D(:,1) = diag(S);
        
        model.U(:,:,2) = eye(nDim,nDim);
        model.U(:,:,3) = eye(nDim,nDim);
        tmp = prod(cand_model.D(:,s))^(1/nDim);
        model.D(:,2) = ones(nDim,1)*tmp;
        model.D(:,3) = ones(nDim,1)*tmp;
        
    case 'diag' 
        model.D = zeros(nDim,3);
        model.D(:,1) = alpha1*cand_model.D(:,m1)+alpha2*cand_model.D(:,m2);
        
        tmp = prod(cand_model.D(:,s))^(1/nDim);
        model.D(:,2) = ones(nDim,1)*tmp;
        model.D(:,3) = ones(nDim,1)*tmp;
        
    case 'spherical'
        model.D = zeros(3,1);
        model.D(1) = alpha1*cand_model.D(m1)+alpha2*cand_model.D(m2);
        model.D(2) = cand_model.D(s);
        model.D(3) = cand_model.D(s);
end

% initialize auxiciliary constants
initLogPxy = gmm_logpxy(X,cand_model);
idx_fixed_comp = setdiff([1:size(cand_model.Mean,2)],[m1 m2 s]);
if length(idx_fixed_comp) > 1
    initLogPxyFixed = logsumexp( initLogPxy(idx_fixed_comp,:));
else
    initLogPxyFixed = initLogPxy(idx_fixed_comp,:);
end

postAdd = logsumexp(initLogPxy([m1 m2 s],:) - repmat(logsumexp(initLogPxy),3,1));

%% run the partial EM steps
logL=[];
for iter=1:options.maxPartialIter
        
  %% Partial E-Step
  % updates posteriors of the three components only
  logPxy = gmm_logpxy(X,model);
  logPost = logPxy - repmat(logsumexp(logPxy),3,1) + repmat(postAdd,3,1 );   
  Alpha = exp( logPost );

  % compute likelihood of the full model
  logPx = logsumexp( [logsumexp(logPxy) ; initLogPxyFixed]);
  logL(iter) = sum( logPx );

  %% Check stopping conditions
  if iter > 1 && logL(iter) - logL(iter-1) < options.minImprovLogL
      break;
  end  
  
  %% Partial M-step 
  % updates only the three components
  model = gmm_ml(X,Alpha,covType,options.minVar);   
  
  % update priors so that sum of new priors equals to sum of their
  % old values
  model.Prior = model.Prior*sum(cand_model.Prior([m1 m2 s]));
  
end 

cand_model.Mean(:,m1)  = model.Mean(:,1);
cand_model.Mean(:,m2)  = model.Mean(:,2);
cand_model.Mean(:,s)  = model.Mean(:,3);
cand_model.logL = logL;

switch cand_model.covType
    case 'full'
        cand_model.U(:,:,m1) = model.U(:,:,1);
        cand_model.U(:,:,m2) = model.U(:,:,2);
        cand_model.U(:,:,s) = model.U(:,:,3);
        
        cand_model.D(:,m1) = model.D(:,1);
        cand_model.D(:,m2) = model.D(:,2);
        cand_model.D(:,s) = model.D(:,3);
        
    case 'diag'
        cand_model.D(:,m1) = model.D(:,1);
        cand_model.D(:,m2) = model.D(:,2);
        cand_model.D(:,s) = model.D(:,3);
        
    case 'spherical'
        cand_model.D(m1) = model.D(1);
        cand_model.D(m2) = model.D(2);
        cand_model.D(s) = model.D(3);
        
end

cand_model.Prior([m1 m2 s]) = ...
    sum(cand_model.Prior([m1 m2 s]))*model.Prior/sum(model.Prior);

return;
% END