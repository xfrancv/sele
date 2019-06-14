% Example of fitting Gaussian Mixture Models by using SMEM and
%  showing SMEM advantage over standard EM.
%

%% SETTINGS
switch 1
    case 1  % GMM with 3 components
        N_SAMPLES = 200;
        COV_TYPE = 'full'; 
%        COV_TYPE = 'diag'; 
%        COV_TYPE = 'spherical'; 
        N_COMP = 3;
        
        MEAN = [2 -2 2; 2 -2 -2];
        COV(:,:,1) = [1 0.2; 0.2 0.5];
        COV(:,:,2) = [0.2 0.1; 0.1 0.5];
        COV(:,:,3) = [0.3 -0.2; -0.2 0.8];
        PRIOR = [0.3 0.4 0.3];
        gmm = gmm_create(MEAN,COV,PRIOR);
 
        X = gmm_samp(gmm,N_SAMPLES);

        % incorrect initial model leading standard EM to a local optimum
        init_model.Mean = [[2;0] [-2;-2] [-3;-3]];
        init_model.U = repmat(eye(2,2),[1 1 3]);
        init_model.D = ones(2,3);
        init_model.covType = COV_TYPE;
        init_model.Prior = [1 1 1]/3;

    case 2 % GMM with 4 components
        N_SAMPLES = 200;
        COV_TYPE = 'full'; 
%        COV_TYPE = 'diag'; 
%        COV_TYPE = 'spherical'; 
        N_COMP = 4;
        
        MEAN = [2 -2 2 -2; 2 -2 -2 2];
        COV(:,:,1) = [1 0.2; 0.2 0.5];
        COV(:,:,2) = [0.2 0.1; 0.1 0.5];
        COV(:,:,3) = [0.3 -0.2; -0.2 0.8];
        COV(:,:,4) = [0.5 -0.2; -0.2 0.9];
        PRIOR = [0.2 0.3 0.2 0.3];
        gmm = gmm_create(MEAN,COV,PRIOR);

        X = gmm_samp(gmm,N_SAMPLES);                
        
        % inccorect initial model leading standard EM to a local optimum
        init_model.Mean = [[2;0] [-2;0] [-2;2] [-2;-2]];
        init_model.U = repmat(eye(2,2),[1 1 4]);
        init_model.D = ones(2,4);
        init_model.covType = COV_TYPE;
        init_model.Prior = [1 1 1 1]/4;

    case 3 % GMM with 4 components
        N_SAMPLES = 500;
        COV_TYPE = 'full'; 
%        COV_TYPE = 'diag'; 
%        COV_TYPE = 'spherical'; 
        N_COMP = 6;
        
        MEAN = [2 -2  2 -2 0  0; ...
                2 -2 -2  2 -4  4];
        COV(:,:,1) = [1 0.2; 0.2 0.5];
        COV(:,:,2) = [0.2 0.1; 0.1 0.5];
        COV(:,:,3) = [0.3 -0.2; -0.2 0.8];
        COV(:,:,4) = [1 -0.4; -0.4 0.2];
        COV(:,:,5) = [2 -0.6; -0.6 1];
        COV(:,:,6) = [1 0.2; 0.2 1];
        PRIOR = [1 1 1 1 1 1]/6;
        gmm = gmm_create(MEAN,COV,PRIOR);

%        X = gmm_samp(gmm,N_SAMPLES);                
        load problem_data;
        
        % inccorect initial model leading standard EM to a local optimum
        init_model.Mean = [[-1;-2] [-.9;-2] [-0.8;-2] [2;-2] [3;-2] [4;-2]];
        init_model.U = repmat(eye(2,2),[1 1 6]);
        init_model.D = ones(2,6);
        init_model.covType = COV_TYPE;
        init_model.Prior = [1 1 1 1 1 1]/6;
end

% Estimate GMM by standard EM
em_est_gmm = gmm_em(X, N_COMP, COV_TYPE,init_model);

% Estimate GMM by SMEM
smem_est_gmm = gmm_smem(X, N_COMP, COV_TYPE,init_model);

% Plot log-likelihood
figure('name','Log-likelihood for SMEM (red) and EM (blue)');
plot(smem_est_gmm.logL,'r'); hold on;
plot(em_est_gmm.logL,'b'); hold on;
xlabel('iteration'); ylabel('logL');

% Plot estimated models
figure('name','True GMM (blue), SMEM (red), EM (green)');
ppatterns(X); hold on; 
pgmm( gmm, 'linestyle','b','comp',1,'values',[0.01],'clabel',0); 
pgmm( smem_est_gmm,'linestyle','r','comp',1,'values',[0.01],'clabel',0);
pgmm( em_est_gmm,'linestyle','g','comp',1,'values',[0.01],'clabel',0);

