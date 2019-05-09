function Model = decision_stump( X, Y, weights )
% DECISION_STUMP Produce classifier thresholding single feature.
%
% Synopsis:
%    Model = decision_stump( X, Y )
%    Model = decision_stump( X, Y, weights )
%
% Description:
%  
% Input:
%   X       [N x M] M N-dimensional training examples.
%   Y       [M x 1] Labels (1 / anything else).
%   weights [M x 1] Weights of training vectors; (by default set to uniform);
% 
% Output:
%  Model [struct] linear classifier.
%   .W  [N x 1] Normal vector of hyperplane.
%   .W0 [1 x 1] Intercept.
%
% Example:
%  help adaboost
%

% About: Statistical Pattern Recognition Toolbox
% (C) 1999-2014, Written by Vojtech Franc and Vaclav Hlavac
% <a href="http://www.cvut.cz">Czech Technical University Prague</a>
% <a href="http://www.feld.cvut.cz">Faculty of Electrical Engineering</a>
% <a href="http://cmp.felk.cvut.cz">Center for Machine Perception</a>

[N,M] = size( X );

if nargin < 3, D = ones(1,M)/M; end;

W = zeros(N,1);
Errors = zeros(N,1);

for i=1:N,
  [x,idx]       = sort( X(i,:) );
  y             = Y(idx);
  D             = weights(idx);
  Sp            = zeros(1,M);
  Sn            = zeros(1,M);
  Sp(y==1)      = D(y==1);
  Sn(y~=1)      = D(y~=1); 
  Sp            = cumsum(Sp); 
  Sn            = cumsum(Sn);
  Tp            = Sp(end);
  Tn            = Sn(end);
  err           = (Sp+Tn-Sn);
  [minerr1,inx1]= min(err);
  [minerr2,inx2]= min(Tp+Tn-err);
  if minerr1 < minerr2,
    W(i) = 1;
    Errors(i) = minerr1;
    if inx1 < M, b(i) = -(x(inx1)+x(inx1+1))*0.5; else b(i)=-(x(inx1)+1); end
  else
    W(i) = - 1;
    Errors(i) = minerr2;
    if inx2 < M,  b(i) = (x(inx2)+x(inx2+1))*0.5; else b(i) = x(inx2)+1; end
  end
end

[dummy,inx]  = min(Errors);
Model.W      = sparse(N,1);
Model.W(inx) = W(inx);
Model.W0     = b(inx);
Model.eval   = @linclassif;

%y = linclass(data.X,model);
%err = sum((y(:)~=data.y(:)).*data.D(:));

return;