function model = fld(X,y)
% FLD Fisher Linear Discriminat.
%
% Synopsis:
%  model = fld(X,y)
%
% Description:
%  This function computes a two-class linear classifier based on the Fisher 
%  Linear Discriminant (FLD). The FLD is the optimal Bayes classifier 
%  with 0/1-loss) provided the examples in the positive and the negative 
%  class are normally distributed and both classes have the same covariance
%  matrix.
%
% Input:
%  X [nDims x nExamples] Training features.
%  y [nExamples x 1] Labels for X (+1 for the positive class and
%      anything else for the negative class).
%
% Output:
%  model [struct] Two-class linear classifier:
%   .W  [nDims x 1] Weights of the linear classifier.
%   .W0 [1x1] Bias of the linear classifier.
%   .separab [1x1] Meassure of class separability.
%
% Example:
%  load('riply_dataset','trn_X','trn_y','tst_X','tst_y');
%  model = fld(trn_X,trn_y);
%  predY = linclassif(tst_X,model);
%  tstError = sum(predY(:) ~= tst_y(:))/length(tst_y) 
%  figure; ppatterns(trn_X,trn_y); pline(model.W,model.W0);
%
% See also 
%

inx1 = find( y == +1);
inx2 = find( y ~= +1);
n1 = length(inx1);
n2 = length(inx2);

m1 = mean( X(:,inx1),2);
m2 = mean( X(:,inx2),2);

S1 = (X(:,inx1)-m1*ones(1,n1))*(X(:,inx1)-m1*ones(1,n1))';
S2 = (X(:,inx2)-m2*ones(1,n2))*(X(:,inx2)-m2*ones(1,n2))';
Sw = S1 + S2;

W = inv(Sw)*(m1-m2);

proj_m1 = W'*m1;
proj_m2 = W'*m2;

model.W = W;
model.W0 = -0.5*(proj_m1+proj_m2);
model.separab = (proj_m1-proj_m2)^2/(W'*Sw*W);
model.eval = 'linclass';

return;

