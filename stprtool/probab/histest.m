function [classCondPos,classCondNeg,priorPos,priorNeg,B] = histest( X, Y, numBins )
% HISTEST Histogram estimate of two-class classification model with univariate features.
%
% Synopsis:
%   [classCondPos,classCondNeg,priorPos,priorNeg,B] = histest( X, Y, numBins )
%
% Description:
%  X and Y contains pairs of samples of independently drawn from random variables
%  distributed according to P*(x,y). Each pair (x,y)=(X(i),Y(i)) is composed 
%  of real number x and a binary label y (+1 / anything else). The function returns 
%  histogram estimate of P*(x,y). In particular it returns:
%  
%   classCondPos(i) = p(x|y==1)  
%   classCondNeg(i) = p(x|y~=1)  
%   priorPos = p(y==1)
%   priorPos = p(y~=1)
%  
%  where x = B(i).
%

if nargin < 3
    numBins = 50;
end

idxPos=find( Y==1);
idxNeg=find( Y~=1);
numPos=length(idxPos);
numNeg=length(idxNeg);

minX = min(X);
maxX = max(X);

B = [minX:(maxX-minX)/numBins:maxX];

priorPos = numPos/(numPos+numNeg);
priorNeg = numPos/(numPos+numNeg);

classCondPos = hist(X(idxPos),B) / length( idxPos);
classCondNeg = hist(X(idxNeg),B) / length( idxNeg);


return;
