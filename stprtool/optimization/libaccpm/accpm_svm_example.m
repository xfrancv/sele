% Example: Training two-class SVM classifier without L2-regularization
% term.

% load training and testing data
load('riply_dataset','Trn','Tst' );

%
Data = risk_svm_init( Trn.X, 1, Trn.Y );
 
% input arguments
Opt.verb   = 1;        % will display progress info
Opt.tolRel = 1e-3;

% reg. parameter; can be 0 if accpm is used; for bmrm lambda > 0
lambda     = 0;   

% call BMRM solver
%[W,stat] = bmrm( Data,@risk_svm,lambda,Opt);  
[W, stat] = accpm( Data, @risk_svm,[],[],100, lambda, Opt );

%
Model = risk_svm_model( Data, W );

% trn and test errors
ypred  = sign2( Model.W'*Trn.X + Model.W0 );
trnErr = sum( ypred(:) ~= Trn.Y(:))/length(ypred)

ypred  = sign2( Model.W'*Tst.X + Model.W0 );
tstErr = sum( ypred(:) ~= Tst.Y(:))/length(ypred)


% 
figure;
ppatterns( Trn.X, Trn.Y);
pline(Model.W,Model.W0);
