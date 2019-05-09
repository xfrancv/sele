% SVMOCAS_NNW training linear two-class SVM classifiers with non-negative weights
%
% Synopsis:
%  [W,W0,stat] = svmocas_nnw(X,X0,y,C,nnw,Method,TolRel,TolAbs,QPBound,BufSize,nExamples,MaxTime)
% 
% Desription:
%  This function trains linear SVM classifier with non-negatie weight by solving
%
%      W,W0 = argmin 0.5*(w'*w+w0^2) + C*sum max( 0, 1-y(i)*(w'*X(:,i)+w0*X0) )
%              w,w0                  i=1:nExamples
%
%      subject to  w(nnw) >= 0
%
%  The standard SVM is obtained if nnw is empty, however, in this case use
%  the function SVMOCAS instead. 
%
%  The training examples are passed in the matrix X where each column is a single
%  feature vector. The matrix X can be one of the following types
%     dense double (default in Matlab)
%     sparse double
%     dense single 
%     dense int8
%  The labels of the training examples must be given as dense vector y whose length
%  equals to the number of columns of the matrix X.
%
% Reference:
%  V. Franc, S. Sonnenburg. Optimized Cutting Plane Algorithm for Large-scale Risk
%  Minimization. Journal of Machine Learning Research. 2009
%    http://jmlr.csail.mit.edu/papers/volume10/franc09a/franc09a.pdf
%
% Input:
%   X [nDim x nExamples] training inputs (dense double, sparse double, dense single, 
%     dense int8 matrix).
%   X0 [1x1] constant coordinate (implicitly) added to all examples;
%     this allows training biased decision rule.
%   y [nExamples x 1] labels (+1/-1).
%   C [1x1]  or [nExamples x 1] C [1x1] is a positive (>0) regularization constant.
%   nnz [P x 1] Indices of weights of W which must be non-negative.
%   Method [1x1] 0..cutting plane; 1..OCAS  (default 1)
%   TolRel [1x1] halts if Q_P-Q_D <= abs(Q_P)*TolRel  (default 0.01)
%   TolAbs [1x1] halts if Q_P-Q_D <= TolAbs  (default 0)
%   QPValue [1x1] halts if Q_P <= QPBpound  (default 0)
%   BufSize [1x1] Initial size of active constrains buffer (default 2000)
%   nExaples [1x1] Number of examples used for training; must be >0 and <= size(X,2).
%     If nExamples = inf then nExamples is set to size(X,2).
%   MaxTime [1x1] halts if time used by solver (data loading time is not counted) exceeds
%    MaxTime given in seconds. Use MaxTime=inf (default) to switch off this stopping condition. 
%
% Output:
%   W [nDim x 1] Paramater vectors of decision rule sign(W'*X+W0)
%   W0 [1x1] Bias term of the decision rule.
%   stat [struct] Optimizer statistics (field names are self-explaining).
%
% Example:
%  % train standard linear SVM classifier and then SVM classifier whose
%  % first weight is non-negative.
%
%  libocasPath = fileparts(which('svmocas'));
%  trn = load([libocasPath '/data/riply_trn.mat'],'X','y'); 
%  svmC = 1;
%  [W,W0,stat] = svmocas(trn.X,1,trn.y,svmC);
%  [W_nn,W0_nn,stat] = svmocas_nnw(trn.X,1,trn.y,svmC,[1]);
%  
%  W
%  W_nn
%

%
% Copyright (C) 2008, 2009, 2012 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
%                                Soeren Sonnenburg, soeren.sonnenburg@first.fraunhofer.de
%
% This program is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public 
% License as published by the Free Software Foundation; 
