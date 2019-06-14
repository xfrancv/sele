%% Example: usage of gradient boosted regression tree
%

%% create training data and plot them
M          = 100;
trnX       = linspace(0,2*pi,M);
noiseLessY = sin(trnX)';
trnY       = noiseLessY + 0.2*randn(size(noiseLessY));

figure;
plot( trnX, trnY, 'xk' ); hold on;
plot( trnX, noiseLessY, 'k');


%% Example 1: fit gbrt regression tree

Opt = [];
Opt.minLeafExamples = 10; 
Opt.minSseImprov    = 1e-6;
Opt.maxDepth        = 3; 
Opt.numTrees        = 50; 
Opt.learningRate    = 0.1; 



Gbrt   = gbrt_train( trnX, trnY, Opt );
predY  = gbrt_pred( trnX, Gbrt );

trnMse = mean( (predY - trnY).^2 )

plot( trnX, predY, 'g');


