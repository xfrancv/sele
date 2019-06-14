%% Example: show how to compute importance of input variables 
%

%% create training data so that only one input variable is correlated 
% with the output while the remaining inputs are just a noise
N            = 10;     % num of inputs 
M            = 100;    % num of training examples
influentTrnX = linspace(0,2*pi,M);
noiseLessY   = sin(influentTrnX)';
trnY         = noiseLessY + 0.2*randn(size(noiseLessY));
dummyTrnX    = rand( N-1, M)*2*pi;

trnX                = zeros(N,M);
idx                 = randperm( N );
influentIdx         = idx(1);
dummyIdx            = idx(2:N);
trnX(dummyIdx,:)    = dummyTrnX;
trnX(influentIdx,:) = influentTrnX;


%% Run gradient boosted reg. tree
Opt = [];
Opt.minLeafExamples = 10; 
Opt.minSseImprov    = 1e-6;
Opt.maxDepth        = 3; 
Opt.numTrees        = 50; 
Opt.learningRate    = 0.1; 

Gbrt  = gbrt_train( trnX, trnY, Opt );
predY = gbrt_pred( trnX, Gbrt );

%% Find the average influence and average usage of the input variables
[importance,usage] = gbrt_importance( Gbrt );

fprintf('\ninfluent input variable: %d\n', influentIdx );
fprintf('var usage importance\n');
fprintf('%2d  %7.4f  %7.4f\n', [[1:N] ; usage'; importance' ]);


%% display dependence of output and the influencial input variable
figure;
plot( influentTrnX, trnY, 'xk' ); hold on;
plot( influentTrnX, noiseLessY, 'k');
plot( influentTrnX, predY, 'g');




