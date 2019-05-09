% Example: L2-regularized multi-class Logistic regression
%

% load training examples
load('fourclassproblem','X','Y');

% input arguments
lambda = 1e-4;      % regularization constant
options.verb = 1;   % display convergence progress

% prepare training data for BMRM
data.X0 = 1;        % add constant feature
data.X =  X;        % training inputs
data.Y = Y;         % training labels

% call BMRM solver
[W,stat]= bmrm(data,@risk_mlogreg,lambda,options);

% extract parameters 
W = reshape(W,size(data.X,1)+(data.X0>0),max(Y)-1);
if data.X0
    W0 = data.X0*W(end,:)';
    W = W(1:end-1,:);
else
    W0 = zeros(max(Y)-1,1);
end

% predict training labels and computes classification error
[dummy,ypred] = max(mlogreg_eval(X,W,W0));
trn_error = sum(ypred ~= Y)/length(Y)

% visualize decision bounary of the classifier
model.W = [zeros(size(X,1),1) W];
model.W0 = [0 ; W0];

figure; 
ppatterns( X, Y );
pclassifier( model, @linclassif );

% EOF