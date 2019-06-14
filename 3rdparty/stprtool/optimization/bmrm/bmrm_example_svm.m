% Example: Train linear SVM by BMRM solver
lambda      = 1e-3;  % regularization parameter

load( 'riply_dataset', 'Trn','Tst');
M = length( Trn.Y ); % number of trn examples

Data      = risk_hinge_init( Trn.X, 1, Trn.Y, ones( M,1)/M );
[W, Stat] = bmrm( Data, @risk_hinge, lambda );
Model     = risk_hinge_model( Data, W );

ypred     = sign( Model.W'*Trn.X + Model.W0 );
trnErr    = mean( ypred ~= Trn.Y )

ypred     = sign( Model.W'*Tst.X + Model.W0 );
tstErr    = mean( ypred ~= Tst.Y )

