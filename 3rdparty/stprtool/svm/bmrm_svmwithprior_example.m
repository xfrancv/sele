%% Example: Train linear SVM classifier with prior on W by BMRM.
%
% It sovles the following problem:
%
%  min  0.5*lambda*W'*W - lambda*W'*Wprior + 1/m sum_{i=1}^m max(0,1-W'*X(:,i)*Y(i))
%   W
%

load('riply_dataset','Trn','Tst');

lambda = 0.001;     % regularization parameter 
x0     = 1;     % linear rule with bias

%%
posCost = 1;
negCost = 1;
Wprior = lambda*[100; 0;0];


%% create data 
Data = risk_svmwithprior_init( Trn.X, x0, Trn.Y, Wprior, posCost, negCost );

%% Call BMRM solver
[W, stat] = bmrm( Data, @risk_svmwithprior, lambda, struct('verb',1) );

%% Create linear classifier from the trained weights W
Model = risk_svm_model( Data, W )

%% training error 
ypred  = sign( Model.W'*Trn.X + Model.W0 );
trnErr = sum( ypred(:) ~= Trn.Y(:) )/numel(Trn.Y)

%% testing error
ypred  = sign( Model.W'*Tst.X + Model.W0 );
tstErr = sum( ypred(:) ~= Tst.Y(:) )/numel(Tst.Y)

%% display training data and decision boundary
figure;
grid on;
ppatterns( Trn.X, Trn.Y);
pclassifier( Model );
