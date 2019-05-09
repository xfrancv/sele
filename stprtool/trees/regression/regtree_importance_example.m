%% Example: show how to compute importance of input variables 
%

%% create training data so that only one input variable is correlated 
% with the output while the remaining inputs are just a noise
N            = 10;     % num of inputs 
M            = 100;    % num of training examples
trnX         = rand(N,M)*2*pi;
idx          = randperm( N );
influentIdx  = idx(1);
[~,idx]      = sort( trnX(influentIdx,:));
trnX         = trnX(:,idx);
noiseLessY   = sin(trnX(influentIdx,:) );
trnY         = noiseLessY + 0.2*randn(size(noiseLessY));


%% Run regression tree
Opt = [];
Opt.minLeafExamples = 5; 
Opt.minSseImprov    = 1e-6;
Opt.maxDepth        = 5; 

Tree   = regtree_train( trnX, trnY, Opt );
predY  = regtree_pred( trnX, Tree );

%% Find the average influence and usage of the input variables
[importance,usage] = regtree_importance( Tree );

fprintf('\ninfluent input variable: %d\n', influentIdx );
fprintf('var usage importance\n');
fprintf('%2d  %7.4f  %7.4f\n', [[1:N] ; usage'; importance' ]);


%% display dependence of output and the influencial input variable
figure;
plot( trnX(influentIdx,:), trnY, 'xk' ); hold on;
plot( trnX(influentIdx,:), noiseLessY, 'k');
plot( trnX(influentIdx,:), predY, 'g');




