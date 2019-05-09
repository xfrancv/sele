function Model = fitlogit(scores, labels, useRegularization)
% FITLOGIT Maximum-likelihood fit of logit function to 1D inputs.
%
% Synopsis:
%  Model = fitlogit(scores, labels)
%  Model = fitlogit(scores, labels, useRegularization )
%
% Description:
%
% Input:
%  scores [N x 1] scores of two class classifier.
%  labels [N x 1] true labels (1...postive class, ~= 1...negative class).
% 
% Output:
%  Model.W  [1x1] 
%  Model.W0 [1x1] 
%
% Example:
%   load( 'riply_dataset', 'Trn');
%   [W,W0] = svmocas(Trn.X, 1, Trn.Y, 1);
%   scores = W'*Trn.X + W0;   
%   Logit = fitlogit( scores, Trn.Y );
%   post1 = logit( [-5:0.1:5], Logit );
%   [classCondPos,classCondNeg,priorPos,priorNeg,B] = histest( scores, Trn.Y, 20 );
%   empPost1 = classCondPos*priorPos ./ (classCondPos*priorPos +classCondNeg*priorNeg);
%   figure;
%   h1=plot(B, empPost1); hold on; 
%   h2=plot(B, 1-empPost1,'r');
%   h3=plot([-5:0.1:5], post1, '--b'); 
%   h4=plot([-5:0.1:5], 1-post1,'r--');
%   legend([h1 h2 h3 h4],'hist p(y=1|x)', 'hist p(y=-1|x)','logit p(y=+1|x)', 'logit p(y=-1|x)');
%
% See also 
%  LOGIT
%



%if nargin > 2,
%  % evaluates log-likelihood (objective function)
%  [L,grad] = sigmoidlogl(arg1,arg2,arg3);
%  varargout{1} = L;
%  varargout{2} = grad;
%  return;
%end

if nargin < 3
    useRegularization = true;
end


idx1 = find(labels==1);
idx2 = find(labels~=1);
N1=length(idx1);
N2=length(idx2);

if useRegularization
  T1=(N1+1)/(N1+2);  % corrected 28. apr 2008; pointed out by Stijn Vanderlooy
  T2=1/(N2+2);
else
  T1=1; 
  T2=0;
end

targets=zeros(N1+N2,1);
targets(idx1)=T1;
targets(idx2)=T2;

% opt=optimset('Display','on','GradObj','on');
opt=optimset('Display','off','GradObj','on');
x0=[1 1];

[x,fval] = fminunc(@logitlogl,x0,opt,targets,scores(:));

Model.W = x(1);
Model.W0 = x(2);
Model.eval = 'logit';

return;


%=======================================================
function [L,grad] = logitlogl(A,targets,outs)
% SIGMOIDLOGL Returns log-likelihood of sigmoid model.
%
% [L,grad] = sigmoidlogl(A,targets,outs)
%
% Description:
%  It evaluates log-likelihood function
%   L = -sum( targets(i)*log(p_i)+(1-targets(i))*log(1-p_i)),
%
% where p_i = 1/(1+exp(A(1)*outs(i)+A(2))) is a sigmoid function.
%

tmp=exp(A(1)*outs+A(2));

p=1./(1+tmp); 

% prevents dividing by 0
inx=find(p==0); p(inx)=1e-12;
inx=find(p==1); p(inx)=1-1e-12;

L = - sum( targets.*log(p)+(1-targets).*log(1-p));

grad(1)=sum(targets.*outs+(outs.*tmp)./(1+tmp) - outs);
grad(2)=sum(targets + tmp./(1+tmp ) - 1);

return;
% EOF