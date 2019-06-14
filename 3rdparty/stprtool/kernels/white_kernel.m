function A = white_kernel(K)
% WHITE_KERNEL
%
% Synopsis:
%  A = white_kernel(K)
%
% Description:
%  TBA!!!
%  It can be used to train kernel machine by an algorithm for training
%  linear rules.
%
% Example:
%  TO UPDATE EXAMPLE !!!
%
%  load('riply_dataset','trn_feat','trn_labels');
%  trn_labels(find(trn_labels~=1)) = -1;
%  width = 0.5;
%  K = kernel(trn_feat,'rbf',width);
%  A = white_kernel(K);
%  trn_feat_new = A*K;
%  [W,W0,stat] = svmocas(trn_feat_new,1,trn_labels,10);
%  model.Alpha = W'*A;
%  model.Alpha = model.Alpha(:);
%  model.b = W0;
%  model.sv.X = trn_feat;
%  model.options.ker = 'rbf';
%  model.options.arg = width;
%  model.fun = 'svmclass';
%  figure;
%  ppatterns(trn_feat,trn_labels);
%  pboundary(model);
%

[U,D] = svd(K);
A=diag(1./sqrt(diag(D)))*U';

return;
