%% Example: usage of gradient boosted regression tree
%

%% create training data and plot them
load('riply_dataset.mat','Trn', 'Tst');
% Trn.X = single(Trn.X);
% Tst.X = single(Tst.X);

%% 
Opt = [];
Opt.minLeafExamples = 10; 
Opt.minObjImprov    = 1e-6;
Opt.maxDepth        = 3; 
Opt.numTrees        = 50; 
Opt.learningRate    = 0.1; 

Gbct   = gbct_train( Trn.X, Trn.Y, Opt, Tst.X, Tst.Y );

[predY,score]  = gbct_pred( Trn.X, Gbct );
trnErr = mean( predY(:) ~= Trn.Y(:) )

predY  = gbct_pred( Tst.X, Gbct );
tstErr = mean( predY(:) ~= Tst.Y(:) )

%%
figure;
ppatterns( Trn.X, Trn.Y ); 
hold on;
pclassifier( Gbct, @gbct_pred );

%%
figure;
h1 = plot( Gbct.trnErr); 
hold on;
h2 = plot( Gbct.valErr);
legend( [h1 h2], 'trnErr','valErr');



