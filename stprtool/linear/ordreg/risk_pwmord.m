function [R,subgrad] = risk_pwmord( Data, W )
% RISK_PWMORD
% 
% Synopsis:
%  [R,subgrad] = risk_pwmord( Data )
%  [R,subgrad] = risk_pwmord( Data, W )
%
% Description:
%

    nExamples = size( Data.X, 2 );

    if nargin < 2, W = zeros( Data.nDims*Data.nZ+Data.nY, 1 ); end

    V0 = W( Data.nDims*Data.nZ+1:end);
    V  = reshape( W(1:Data.nDims*Data.nZ), Data.nDims, Data.nZ );

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

    alpha = zeros(Data.nZ,nExamples);
    for i = 1 : nExamples, 
        alpha(:,i) = Data.A(:,ypred(i)) - Data.A(:,Data.Y(i)); 
    end
    subgrad1 = Data.X*sparse(alpha)';

    subgrad2 = zeros(Data.nY,1);
    for y = 1 : Data.nY, 
        subgrad2(y) = sum(ypred==y) - sum(Data.Y==y); 
    end

    subgrad = [subgrad1(:) ; subgrad2(:)] / nExamples;
    R       = R / nExamples;

end
% EOF