function [y,fval]=chain_maxsum(G,Q)
% CHAIN_MAXSUM Solve MAXSUM problem on a chain neighbourhood.
%
% Synopsis:
%    [y,fval]=chain_maxsum(G,Q) 
%
% Description:
%  It solves
%   max_(y1,...,yN) [ sum_{i=1}^N Q(y_i,i) + sum_{i=1}^{N-1} G(y_i,y_{i-1},i) ] 
%  
% Examples:
%  %% Finding MAP sequence on HMM
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

    
Y = zeros(size(Q)); 
F = zeros(size(Q));
[nY,nT] = size(Q);
if size(G,3) == 1
    G = repmat(G,[1 1 nT-1]);
end


F(:,1) = Q(:,1);

for t=2:nT,
  for y=1:nY,
    tmp = -inf;
    for yy=1:nY,
      if tmp < G(yy,y,t-1)+F(yy,t-1),
         tmp = G(yy,y,t-1)+F(yy,t-1);
         Y(y,t) = yy;
      end
    end 
    F(y,t) = tmp+Q(y,t);
  end
end

y = zeros(1,nT);
[fval,y(nT)] = max(F(:,nT));

for t=nT:-1:2,
  y(t-1) = Y(y(t),t);
end
  
return;
% EOF