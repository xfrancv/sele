function m = hmm_margin(Y,prior,trans,emisX)
% HMM_MARGIN Marginalize over given set of states.
%
% Synopsis:
%   m = hmm_margin(Y,prior,trans,emisX)
%
% Description:
%   Y [nY x nT] zero-one matrix indicating allowed states
%   prior [nY x 1]  prior(y) = P_1(y)  prior of the first state
%   trans [nY x nY] trans(y,yy) = P_i(yy|y)
%   emisX  [nY x nT] emisX(y,i) = P_i(X(i)|y)
%
% Example:
%   TBA: example needs to be updated; corresponds to the old version of this funtion
%   trans = [0.8,0.2;
%            0.2,0.8];
%   prior = [0.4 0.6];
%   Y = ones(2,10);   
%   m = hmm_margin(Y,trans,prior)
%   Y1=Y; Y1(1,end) = 0;
%   m = hmm_margin(Y1,trans,prior)
%   Y2=Y; Y2(2,end) = 0;
%   m = hmm_margin(Y2,trans,prior)

[nY,nT] = size(Y);
 
if ndims(trans) < 3
    trans = repmat(trans,[1 1 nT-1]);
end

if nargin < 4
    emisX = ones(nY,nT);
end

f = ones(nY,1);
for t=nT:-1:2
    ff = zeros(nY,1);
    idx = find(Y(:,t));
    for yy=idx(:)'
        ff = ff + f(yy)*trans(:,yy,t-1)*emisX(yy,t);
    end
    f=ff;
end

idx = find(Y(:,1));
prior = prior(:).*emisX(:,1);
m = ff(idx)'*prior(idx);

return;