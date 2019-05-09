function [R,subgrad,Data] = risk_genlinbias(Data,W)
% RISK_GENLINABS
% 
% Synopsis:
%  [R,subgrad,data] = risk_genlinbias(Data)
%  [R,subgrad,data] = risk_genlinbias(Data,W)
%
% Description:
%

[nDims,nExamples] = size( Data.X );

if nargin < 2
   W = zeros( nDims*Data.nZ+Data.nY, 1 );
end

V0 = W(nDims*Data.nZ+1:end);
V = reshape( W(1:nDims*Data.nZ), nDims, Data.nZ );

P = V*Data.A;

score = P' * Data.X + repmat(V0,1,nExamples);
Delta = abs( repmat([1:Data.nY]',1,nExamples) - repmat( Data.Y(:)',Data.nY, 1)) ;
score = Delta + score;

linearTermOfRisk = 0;
for y=1:Data.nY
    idx = find(Data.Y==y);
    linearTermOfRisk = linearTermOfRisk + sum(P(:,y)'*Data.X(:,idx)) + length(idx)*V0(y);
end

[losses,ypred] = max(score);

R = sum(losses)-linearTermOfRisk;

subgrad = zeros(nDims,Data.nZ);
subgrad0 = zeros(Data.nY,1);

alpha = zeros(Data.nZ,nExamples);

for i=1:nExamples
    tmp = Data.A(:,ypred(i)) - Data.A(:,Data.Y(i));    
    alpha(:,i) = tmp(:);    
end
subgrad = Data.X*alpha';

for y=1:Data.nY
    subgrad0(y) = sum(ypred==y) - sum(Data.Y==y);
end

subgrad = [subgrad(:);subgrad0(:)]/nExamples;
R = R/nExamples;

return;
% EOF