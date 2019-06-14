%% Example: computes expected dependecy of the output on a single input
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


%% Run gradient boosted reg. tree
Opt = [];
Opt.minLeafExamples = 10; 
Opt.minSseImprov    = 1e-6;
Opt.maxDepth        = 3; 
Opt.numTrees        = 50; 
Opt.learningRate    = 0.1; 

Gbrt  = gbrt_train( trnX, trnY, Opt );
predY = gbrt_pred( trnX, Gbrt );


%inVarIdx = 1;
inVarIdx = influentIdx;

x = linspace( 0, 2*pi, 100);
y = gbrt_marginals( Gbrt, inVarIdx, x );



%% display dependence of output on the individual vairables
figure;
plot( x, y, '-b' ); hold on;
hold on;
plot( trnX(influentIdx,:), noiseLessY, 'r');
plot( trnX(influentIdx,:), trnY, 'xk' ); 
plot( trnX(influentIdx,:), predY, 'g');
grid on;

