function [R,subgrad] = risk_logreg( Data, W )
% RISK_LOGREG evaluates risk term of the logistic regression.
% 
% Synopsis:
%  [R,subgrad] = risk_logreg( Data )
%  [R,subgrad] = risk_logreg( Data,W )
%
% Description:
%  If Data.X0 ~= 0 then the risk is defined as
%
%                                              m
%  R(W) = 0.5*lambda*norm(W(1:end-1))^2 + 1/m sum log(1+exp( Y(i)*(W(1:end-1)'* X(:,i)+ X0*W(end))))
%                                             i=1
%
%  If Data.X0 == 0 then
%                                m
%  R(W) = 0.5*lambda*W'*W + 1/m sum log(1+exp( Y(i)*W'* X(:,i) ))
%
%  The features/labels (X,Y) are elements of the structure Data.
%                               i=1
% 
%  This function returns value of R(W) and a subgradient of R(W) at point W.
%  
%  Calling the function with only one arguments means that W=0.
%

if nargin < 2        
    % evaluates risk and subgradient at W = 0

    nExamples = size(Data.X,2);
    
 %   R = sum(log(1+ones(1,nExamples)))/nExamples;
    R = log(2);
    
    if Data.X0 ~= 0

        % contant feature added
        subgrad = -[Data.X*Data.Y(:);Data.X0*sum(Data.Y(:))]/(2*nExamples);
    else
        subgrad = -Data.X*Data.Y(:)/(2*nExamples);
    end
else
        
    nExamples = size( Data.X, 2 );
        
    if Data.X0 ~= 0

        % contant feature added
        proj    = exp(-(W(1:end-1)'*Data.X+Data.X0*W(end)).*Data.Y(:)');
        R       = 0.5*Data.lambda*norm(W(1:end-1))^2 + sum(log(1+proj))/nExamples;
        subgrad = [Data.lambda*W(1:end-1);0] - ...
                  [Data.X*((proj(:).*Data.Y(:))./(1+proj(:))); ...
                   Data.X0*sum((proj(:).*Data.Y(:))./(1+proj(:)))]/nExamples;
    else
        proj    = exp(-(W'*Data.X).*Data.Y(:)');
        R       = 0.5*Data.lambda*norm(W)^2 + sum(log(1+proj))/nExamples;
        subgrad = Data.lambda*W - Data.X*((proj(:).*Data.Y(:))./(1+proj(:)))/nExamples;
    end
end
   
return;
% EOF