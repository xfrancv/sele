function [R,subgrad] = risk_svmwithprior( Data, W )
% RISK_SVMWITHPRIOR L1-hinge loss for two-class linear classifier.
% 
% Synopsis:
%  [R,subgrad,data] = risk_svmwithprior(data)
%  [R,subgrad,data] = risk_svmwithprior(data,W)
%
% Description:
%  Let the risk term be defined as
%
%  R(W) = 1/m sum_{i=1}^m max(0, 1-data.y(i)*(W'*[data.X(:,i);data.X0]))
%
%  This function returns value R and subgradient SUBGRAD of the 
%  risk R(W) at W.
%

    [nDim, nExamples] = size( Data.X );

    if nargin < 2, W = zeros(nDim + double(Data.X0==1),1); end

    if Data.X0 == 0
        proj    = Data.X'*W;
        score   = abs(Data.Y) - proj.*Data.Y;    
        idx     = find( score > 0);                
        R       = sum( score(idx) ) / nExamples - W'*Data.Wprior;
        subgrad = -Data.X(:,idx)*Data.Y(idx) / nExamples - Data.Wprior;
    else
        proj    = Data.X'*W(1:end-1) + W(end)*Data.X0;
        score   = abs(Data.Y) - proj.*Data.Y;    
        idx     = find( score > 0);                
        R       = sum( score(idx) ) / nExamples - W'*Data.Wprior;
        subgrad = [-Data.X(:,idx)*Data.Y(idx) ; -sum(Data.Y(idx))*Data.X0] / nExamples - Data.Wprior;
    end
        
end
% EOF