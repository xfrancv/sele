%% Example: training two-class Logistic regression classifier
%

load( 'riply_dataset', 'Trn', 'Tst' );

Opt.maxIter = 100;
Opt.eps     = 1.e-5;
Opt.m       = 5;
Opt.verb    = 1;

lambda = 0;  % strength of quadratic regularization
X0     = 1;  % biased linear rule

Data   = risk_logreg_init( Trn.X, X0, Trn.Y, lambda );
W      = lbfgs( Data, 'risk_logreg', [], Opt );
Model  = risk_logreg_model( Data, W );

figure;
ppatterns( Trn.X, Trn.Y );
pline( Model.W, Model.W0);
grid on;

ypred  = linclassif( Trn.X, Model );
trnErr = mean( ypred(:) ~= Trn.Y(:))

ypred  = linclassif( Tst.X, Model );
tstErr = mean( ypred(:) ~= Tst.Y(:))