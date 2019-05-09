% Example: Train linear SVM classifier by OCA solver.
%

% prepare data  
x0           = 1;     % linear rule with bias
lambda       = 1e-4;  % regularization parameter 
Opt.bufSize  = 1000;
Opt.useCplex = 0;

%%
load('riply_dataset','Trn','Tst');
M = length( Trn.Y );

Data  = risk_hinge_init( Trn.X, 1, Trn.Y, ones( M,1)/M );
[W,S,C]     = oca( Data, @riskls_hinge, lambda, [], [], [], Opt );
Model = risk_hinge_model( Data, W );

ypred  = sign( Model.W'*Trn.X + Model.W0 );
trnErr = mean( ypred(:) ~= Trn.Y(:) )

ypred  = sign( Model.W'*Tst.X + Model.W0 );
tstErr = mean( ypred(:) ~= Tst.Y(:) )
