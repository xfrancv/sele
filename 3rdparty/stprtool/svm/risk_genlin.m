function [R,subgrad,Data] = risk_genlin(Data,W)
% RISK_GENLINABS
% 
% Synopsis:
%  [R,subgrad,data] = risk_genlin(Data)
%  [R,subgrad,data] = risk_genlin(Data,W)
%
% Description:
%

[nDims,nExamples] = size( Data.X );

if nargin < 2
    W = zeros( nDims*Data.nZ, 1 );
   
    % no constant bias by default
    if ~isfield(Data,'bias')
        Data.bias = zeros(Data.nY,1);
    end
    
    % use 0/1-loss by default
    if ~isfield(Data,'loss')         
        Data.loss = ones(Data.nY,Data.nY) - eye(Data.nY,Data.nY);
    end
    
    Data.lossMatrix = zeros(Data.nY,nExamples);
    for y=1:Data.nY
        idx = find(Data.Y==y);
        Data.lossMatrix(:,idx) = repmat(Data.loss(:,y),1,length(idx));
    end      
   
end

W = reshape( W, nDims, Data.nZ );

V = W*Data.A;

score = V' * Data.X;
%%Delta = abs( repmat([1:Data.nY]',1,nExamples) - repmat( Data.Y(:)',Data.nY, 1)) ;
score = Data.lossMatrix + score + repmat( Data.bias(:), 1, nExamples );

linearTermOfRisk = 0;
for y=1:Data.nY
    idx = find(Data.Y==y);
    linearTermOfRisk = linearTermOfRisk + sum(V(:,y)'*Data.X(:,idx)) + length(idx)*Data.bias( y );
end

[losses,ypred] = max(score);

R = sum(losses)-linearTermOfRisk;

subgrad = zeros(nDims,Data.nZ);

alpha = zeros(Data.nZ,nExamples);

for i=1:nExamples
    tmp = Data.A(:,ypred(i)) - Data.A(:,Data.Y(i));
    
    alpha(:,i) = tmp(:);    
end
subgrad = Data.X*alpha';

subgrad = subgrad(:)/nExamples;
R = R/nExamples;

return;
% EOF