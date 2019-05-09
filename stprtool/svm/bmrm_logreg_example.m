% Example: L2-regularized Logistic regression

% load training and testing examples 
load('riply_dataset','trn','tst');

% input arguments
lambda = 1e-4;      % regulrization constant 
Options.verb = 1;   % display convergence progress

% prepare training data for BMRM
Data = trn;
Data.X0 = 1;        % add constant feature -> biased linear rule

% call BMRM solver
[W,Stat]= bmrm(Data,@risk_logreg,lambda,Options);

if Data.X0 == 0
    W0 = 0;
else
    W0 = W(end);
    W = W(1:end-1);
end

% compute training and testing error
ypred = sign(W'*trn.X + W0);
trn_error = sum(ypred(:)~=trn.Y(:))/length(trn.Y)

ypred = sign(W'*tst.X + W0);
tst_error = sum(ypred(:)~=tst.Y(:))/length(tst.Y)

% visualize bounary of the trained linear classification rule
figure;
ppatterns(trn.X,trn.Y);
pline(W,W0);