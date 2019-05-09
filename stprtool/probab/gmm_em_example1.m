% Example: Using EM to estimate GMM in 1D.

%% SETTINGS
nSamples = 200;
nComp    = 3;

% Set up ground truth GMM 
means  = [-2 0 2];
vars   = [0.3 0.1 0.2];
priors = [0.3 0.5 0.2];

% Generate examples from ground truth GMM
Gmm = gmm_create( means,vars,priors);
X   = gmm_samp( Gmm,nSamples);

% Estimate GMM by EM from the generated examples
EstGmm = gmm_em( X, nComp );

% Vizualize the results
figure('name','Log-likelihood');
plot(EstGmm.logL); xlabel('iteration'); ylabel('logL');

figure('name','True GMM (blue), estimated by EM (red)');
plot(X,0,'ok'); hold on; 
pgmm( Gmm, 'linestyle','b'); 
pgmm( EstGmm,'linestyle','r','comp',1);

