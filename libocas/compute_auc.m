% COMPUTE_AUC computes area under ROC.
%
% Synopsis:
%  auc = compute_auc(score,trueLabels)
% 
% Input:
%   score [N x 1 (double)] score of two-class classifier; positive class has 
%       score >= 0 while negative class has score < 0 
%   trueLabels [N x 1 (double)] positive class 1; negative class ~= 1
% Output: 
%   auc [1x1] Area Under ROC.
%
% Example:
%   help svmocas
%
