% Example: Using EM to estimate GMM in 2D.
%

%% SETTINGS
nSamples = 200;
covType  = 'full'; 
nComp    = 2;

% Define ground truth GMM
means       = [1 -1 ;1 -1 ];
covs(:,:,1) = [1 0.2; 0.2 0.1];
covs(:,:,2) = [0.3 0.1; 0.1 1];
priors      = [0.3 0.7];

% Generate examples
Gmm = gmm_create( means, covs, priors);
X   = gmm_samp( Gmm, nSamples);

% Estimate GMM from examples by
EstGmm = gmm_em( X, nComp, covType);

% Visualiziation of results
figure('name','Log-likelihood');
plot(EstGmm.logL); xlabel('iteration'); ylabel('logL');

figure('name','True GMM'); 
ppatterns(X); hold on; pgmm( Gmm );

figure('name','GMM estimated by EM');
ppatterns(X); hold on; pgmm( EstGmm );

figure('name','True GMM (blue), estimated (red)');
ppatterns(X); hold on; 
pgmm( Gmm, 'linestyle','b'); 
pgmm(EstGmm,'linestyle','r');

