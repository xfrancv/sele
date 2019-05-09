function [M,FL,BL]=chain_maxmarginals(G,Q)
% CHAIN_MAXMARGINALS Compute max-marginals for chain energy.
%
% Synopsis:
%    M = chain_maxmarginals(G,Q) 
%
% Description:
%  It solves
%   M(yy,t) = max_(y1,...,yt=yy,...yN) [ sum_{i=1}^N Q(y1,i) + sum_{i=1}^{N-1} G(y_i,y_{i-1},i) ] 
%  
% Examples:
%  %% Finding MAP sequence of HMM
%  N = 10;
%  trans = [0.6,0.4;0.7,0.3];
%  emis =  [1/6,  1/6,  1/6,  1/6,  1/6,  1/6; 1/10, 1/10, 1/10, 1/10, 1/10, 1/2];
%  [seq, states] = hmmgenerate(N,trans,emis)
%  estimatedStates1 = hmmviterbi(seq,trans,emis)
%
%  G = log(trans);
%  Q = zeros(2,N); Q(:,1) = log(trans(1,:)');
%  for i=1:N
%    Q(:,i) = Q(:,i) + log(emis(:,seq(i)));
%  end
%  [estimatedStates2,fval]=chain_maxsum(G,Q)
%
%  M=chain_maxmarginals(G,Q);
%  max(M)
%

B=zeros(size(Q));
BL=zeros(size(Q));
[nY,nT] = size(Q);

B(:,nT) = Q(:,nT);
for t=nT-1:-1:1
    for y=1:nY
        [B(y,t),BL(y,t)] = max(Q(y,t) + B(:,t+1)+G(y,:)');
    end
end

F=zeros(size(Q));
FL=zeros(size(Q));
F(:,1) = Q(:,1);
for t=2:nT
    for y=1:nY
        [F(y,t),FL(y,t)] = max(Q(y,t) + F(:,t-1)+G(:,y));
    end
end

M = F + B - Q;
  
return;
% EOF