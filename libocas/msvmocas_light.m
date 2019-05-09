% MSVMOCAS_LIGHT OCAS solver for training multi-class linear SVM classifier from SVM^light file.
% 
%
% Synopsis:
%  [W,stat] = msvmocas_light(dataFile,C,Method,TolRel,TolAbs,QPBound,BufSize,nExamples,MaxTime,verb)
%
% Desription:
%  This function trains multi-class linear SVM classifier by solving
%
%      W^* = argmin 0.5*sum_y (W(:,y)'*W(:,y)) + C*  sum     max( (y~=y(i)) + (W(:,y) - W(:,y(i))'*X(:,i))
%              W                                   i=1:nData   y
%
%  The function loads examples from dataFile in SVM^light format.
%
% Reference:
%  V. Franc, S. Sonnenburg. Optimized Cutting Plane Algorithm for Large-scale Risk
%  Minimization. Journal of Machine Learning Research. 2009
%    http://jmlr.csail.mit.edu/papers/volume10/franc09a/franc09a.pdf
%
% Input:
%   dataFile [string] path to file with training examples in SVM^light format.
%   y [nExamples x 1] labels; intgers 1,2,...nY
%   C [1x1] regularization constant
%   Method [1x1] 0..cutting plane; 1..OCAS  (default 1)
%   TolRel [1x1] halts if Q_P-Q_D <= abs(Q_P)*TolRel  (default 0.01)
%   TolAbs [1x1] halts if Q_P-Q_D <= TolAbs  (default 0)
%   QPValue [1x1] halts if Q_P <= QPBpound  (default 0)
%   BufSize [1x1] Initial size of active constrains buffer (default 2000)
%   nExamples [1x1] Number of training examplesused for training; must be >0 and <= size(X,2).
%     If nExamples = inf then nExamples is set to size(X,2).
%   MaxTime [1x1] halts if time used by solver (data loading time is not counted) exceeds
%    MaxTime given in seconds. Use MaxTime=inf (default) to switch off this stopping condition. 
%   verb [1x1] if non-zero then prints some info; (default 1)
%
% Output:
%   W [nDim x nY] Paramater vectors of decision rule; [dummy,ypred] = max(W'*x)
%   stat [struct] Optimizer statistics (field names are self-explaining).
%
% Example:
%  % training
%  libocasPath = fileparts(which('svmocas'));
%  svmC = 1; 
%  [W,stat] = msvmocas_light([libocasPath '/data/example4_train.light'],svmC);
%
%  % classifying test examples
%  [score,trueLabels] = linclassif_light([libocasPath '/data/example4_test.light'],W,[]);
%  [dummy,ypred] = max(score);
%  sum(ypred~=trueLabels)/length(trueLabels)
% 

%
% Copyright (C) 2008 Vojtech Franc, xfrancv@cmp.felk.cvutr.cz
%                    Soeren Sonnenburg, soeren.sonnenburg@first.fraunhofer.de
%
% This program is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public 
% License as published by the Free Software Foundation; 
% Version 3, 29 June 2007
