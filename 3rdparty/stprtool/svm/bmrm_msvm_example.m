% Example: Learning multi-class linear classifier by regularized risk
%    minimization. It shows how to use different kind of loss functions.
%   

% load training examples
load('fourclassproblem','X','Y');

lambda = 1e-3;      % regularization constant
X0     = 1;         % constant feature

% define loss
nY   = max(Y);
Y1   = repmat([1:nY],nY,1); 
Y2   = repmat([1:nY]',1,nY);

switch 1
    case 1 % zero-one loss                      
        loss = double(Y1~=Y2);  
        
    case 2 % Mean absolute deviation
        loss = abs(Y1-Y2);      
    
    % recognizing class 1 is important: 
    % Penalty for assigning object with label 1 to another class 
    % will be 100 while other misclassifications have penaly 1.
    case 3              
        loss = double(Y1~=Y2);
        loss(2:end)=100;                
end

% create data 
Data = risk_msvm_init( X, X0, Y, loss );

% call BMRM solver
[W,stat] = bmrm( Data, @risk_msvm, lambda );

% Create model
Model = risk_msvm_model( Data, W );

% Training error
ypred  = linclassif( X, Model );
trnErr = sum( ypred(:) ~= Y(:) ) / numel( Y ) 

% visualize decision bounary  and training data
figure; 
ppatterns( X, Y );
pclassifier( Model, @linclassif );
