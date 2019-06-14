% SVMLIGHT_LINCLASS classifies examples in SVM^light file by linear rule.
%
% Synopsis:
%  [score,trueLabels] = linclassif_light(dataFile,W,[])
%  [score,trueLabels] = linclassif_light(dataFile,W,W0)
%  [score,trueLabels] = linclassif_light(dataFile,W,W0,verb)
% 
% Input:
%  dataFile [string] File with examples stored in the SVM^light format.
%  W [nDims x nModels] Parameter vectors of nModels linear classifiers.
%  W0 [nModels x 1] Bias of decision rule. If W0 is empty then W0 = 0 is used.
%  verb [1x1] If ~= 0 then prints info (default 0).
%
% Output:
%  score [nModels x nExamples] score(i,j) = W(:,i)'*X_j + W0(i)
%  trueLabels [nExamples x 1] labels from the dataFile
%
% Example:
%  help svmocas_light
%
