% Example: Train two-class linear SVM classifier traned by minimizing
%
%   F(W) = 0.5*lambda*norm(W) + sum( C.* max(0,1-(X'*W).*Y) )
%
%   where lambda [1x1] is a constant, C [M x 1] are cost factors, 
%   X [N x M] features and Y [M x 1] are labels (+1/-1).
%


lambda = 1e-3;  % regularization parameter 
Cpos   = 1;     % positive class cost factor
Cneg   = 1;     % negative class cost factor
x0     = 1;     % linear rule with bias (added constant variable)

Opt.maxIter = inf;
Opt.bufSize = 1000;
Opt.tolRel  = 0.001;

% create data 
load('riply_dataset','Trn','Tst');

C    = ( Cpos*double(Trn.Y==1) + Cneg*double(Trn.Y~=1) )/length( Trn.Y);
Data = risk_hinge_init( Trn.X, x0, Trn.Y, C );
        
% Call solver 
switch 'oca'
    case 'bmrm'
        % BMRM solver [Teo et al. XXX]
        [W, Stat] = bmrmconstr( Data, @risk_hinge, lambda, [], [], [], Opt );
        
    case 'oca'
        % OCA solver [Franc et al 2009]
        [W, Stat] = oca( Data, @riskls_hinge, lambda, [], [], [], Opt );
end

% Create linear classifier from the trained weights W
Model = risk_hinge_model( Data, W )

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
