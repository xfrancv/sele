function p = logit(X,Model)
% LOGIT Evaluates logit function.
%
% Synopsis:
%  p = logit(X,Model)
%
% Description:
%   p = 1./(1+exp(Model.W'*X + Model.W0)'
%
% Input:
%  X [D x N] Inputs.
%  Model [struct] Parameters of logit function:
%    .W [D x 1] 
%    .W0 [1 x 1]
%
% Output:
%  p [N x 1] Values of the logit function.
%
% See also 
%  FITLOGIT
%

p = 1./(1+exp(Model.W'*X + Model.W0))';

return;
% EOF