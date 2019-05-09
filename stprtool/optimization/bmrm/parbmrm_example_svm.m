% Example: Train linear SVM by BMRM solver
lambda   = 1e-3;  % regularization parameter
nThreads = 2;

load( 'riply_dataset', 'Trn','Tst');
M = length( Trn.Y ); % number of trn examples

Opts.tolRel = 0.000001;
Opts.verb = 1;
Opts.useParfor = 1;
Opts.useCplex = 0;

Data = [];
idx  = randperm(M);
from = 1;
for p = 1 : nThreads
    to      = round( p*M/nThreads);
    Data{p} = risk_hinge_init( Trn.X(:,idx(from:to)), 1, Trn.Y(idx(from:to)), ones( M,1)/M );
    from    = to + 1;
end

[W, Stat] = parbmrm( Data, @risk_hinge, lambda, Opts );
Model     = risk_hinge_model( Data, W );

ypred     = sign( Model.W'*Trn.X + Model.W0 );
trnErr    = mean( ypred ~= Trn.Y )

ypred     = sign( Model.W'*Tst.X + Model.W0 );
tstErr    = mean( ypred ~= Tst.Y )

