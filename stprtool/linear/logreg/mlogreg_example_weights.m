%% Example: training multi-class Logistic regression classifier
%  when the log-likelihoods of individual examples have different weights
%

load( 'kolac', 'Trn', 'Tst' );

Opt.maxIter = 100;
Opt.eps     = 1.e-5;
Opt.m       = 5;
Opt.verb    = 1;

lambda = 0;  % strength of quadratic regularization
X0     = 1;  % biased linear rule

% examples of class 1 and 4 have 10 times higher weight than other classes
weights = ones( size( Trn.Y(:) ));
weights( find( Trn.Y == 1) ) = 10;
weights( find( Trn.Y == 4) ) = 10;
weights = weights / sum( weights );

Data   = risk_mlogreg_init( Trn.X, X0, Trn.Y, lambda, weights );
W      = lbfgs( Data, 'risk_mlogreg', [], Opt );
Model  = risk_mlogreg_model( Data, W );

figure;
ppatterns( Trn.X, Trn.Y );
pclassifier( Model, @linclassif );
grid on;

ypred  = linclassif( Trn.X, Model );
trnErr = mean( ypred(:) ~= Trn.Y(:))

ypred  = linclassif( Tst.X, Model );
tstErr = mean( ypred(:) ~= Tst.Y(:))