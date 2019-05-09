function [y,score] = kernelclassif( X, Model)
% KERNELCLASSIF Kernel classifier.
%
% Synopsis:
%  [y,score] = kernelclassif( X, Model )
%
% Description:
%  This function implements two-class and multi-class kernel classifier.
%
%  Let score = kernelmap( X, model) be values of score functions obtained 
%  by projecting input objects X onto vectors in kernel Hilber space.
%  See HELP KERNELMAP for more info on how the score function is computed. 
%
%  If size(model.Alpha,1)==1 then the two-class classifier is applied
%    [dummy, y] = max( [score; -score] )
%
%  If size(model.Alpha,1) > 1 then the multi-class classifier is used
%    [dummy, y] = max( score )
%           
% Input:
%  X [array of N objects] Input objects to be classified.
%  model [struct] Kernel classifier:
%   .Alpha [nScoreFcs x nSV] Multipliers associated to suport vectors.
%   .bias [nScoreFcs x 1] Biases.
%   .SV [array of nSV objects] Support vectors.
%   .kernel_name [string] Kernel identifier. See HELP KERNEL for more info.
%   .kernel_args [...] Kernel arguments.
%
% Output:
%  y [nObjects x 1] Predicted labels (1,...nY).
%  score [nY x nObjects] Score functions.
%
% Example:
%  TBA
% 
% See also 
%  TBA
%

score = kernelmap(X, Model);

if size(score,1) == 1,
  %% Two-class classifier
  y = 2*(score >= 0) - 1;

else  
  %% Multi-class classifier
  [dummy,y] = max( score );
end

return;
% EOF

