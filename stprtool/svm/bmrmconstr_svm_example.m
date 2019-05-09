% Example: Train linear SVM classifier by BMRMCONSTR. 
%  Allows to set prior on the solution vector W via set of linear
%  enequalities A*w >= b.
%

load('riply_dataset','Trn','Tst');

% prepare data  
x0   = 1;     % linear rule with bias
Data = risk_svm_init( Trn.X, x0, Trn.Y );
        
%% Using constraints A*w >= b. ensure that weights are in a box [-5 5]
A = [1 0 0; -1 0 0; 0 1 0; 0 -1 0];
b = [-1; -1; -1; -1]*5;

% A = [-1 0 0; 0 -1 0];
% b = 2*[1; 1];

%% to switch off constraints just use
%  A = [];
%  b = [];

% Call BMRM solver
lambda       = 1e-4;  % regularization parameter 
Opt.bufSize  = 10;
Opt.useCplex = 1;
[W, Status, Cpm] = bmrmconstr( Data, @risk_svm, lambda, A, b, [], Opt );

% Create linear classifier from the trained weights W
Model = risk_svm_model( Data, W )

% training error 
ypred  = sign( Model.W'*Trn.X + Model.W0 );
trnErr = sum( ypred(:) ~= Trn.Y(:) )/numel(Trn.Y)

% testing error
ypred  = sign( Model.W'*Tst.X + Model.W0 );
tstErr = sum( ypred(:) ~= Tst.Y(:) )/numel(Tst.Y)

% display training data and decision boundary
figure;
grid on;
ppatterns( Trn.X, Trn.Y);
pclassifier( Model );
