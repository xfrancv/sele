function ReducedModel = rsrbf( Model, newNumSV, Options )
% RSRBF Redused Set Method for RBF kernel expansion.
%
% Synopsis:
%  ReducedModel = rsrbf( Model, newNumSV )
%  ReducedModel = rsrbf( Model, newNumSV, Options )
%
% Description:
%  It finds kernel RBF expansion with newNumSV support vectors 
%  which best approximates the input kernel expansion.
%
%  The function can be used to simplify two-class RBF-SVM classifier
%  in which case the kernel expansion is the decision function.
%  See the example below.
%    
% Input:
%  Model [struct] Kernel expansion:
%   .Alpha [ nSV x 1] Weight vector.
%   .SV [nDim x nSV] Vectors defining the expansion.
%   .kernelName [string] Must be set to 'rbf'.
%   .kernelArgs [1x1]    Gaussian width for RBD (see 'help kernel').
%
%  newNumSV [ 1x1] Number of support vectors in the output model.
%
%  Options [struct]
%    .maxDev [ 1x1] Maximal distance between the original and new decision
%                  vector. Default 1e-6.
%    .verb   [1x1] defualt true;
% 
% Output:
%  ReducedModel [struct] Model with reduced number of support vectors.
%
% Example:
% 
%   load('riply_dataset','Trn','Tst');
%   svmC = 100; rbfWidth = 0.5;
%   SvmClassif = svmocas_kernel(Trn.X,1,Trn.Y,svmC,'rbf',rbfWidth);
%   trnErr1 = sum(kernelclassif( Trn.X, SvmClassif ) ~= Trn.Y(:)')/length(Trn.Y)
%   tstErr1 = sum(kernelclassif( Tst.X, SvmClassif ) ~= Tst.Y(:)')/length(Tst.Y)
%
%   newNumberOfSupportVectors = 15;
%   ReducedSvmClassif = rsrbf( SvmClassif, newNumberOfSupportVectors );
%   trnErr2 = sum(kernelclassif( Trn.X, ReducedSvmClassif ) ~= Trn.Y(:)')/length(Trn.Y)
%   tstErr2 = sum(kernelclassif( Tst.X, ReducedSvmClassif ) ~= Tst.Y(:)')/length(Tst.Y)
%
%   figure;
%   ppatterns( Trn.X, Trn.Y);
%   h1=ppatterns( ReducedSvmClassif.SV, [],'BigCircles');
%   h2=pclassifier( SvmClassif, [], struct('LineSpec','r') );
%   h3=pclassifier( ReducedSvmClassif, [], struct('LineSpec','b') );
%   legend([h1.points h2 h3],'New support vectors','Original rule','Reduced rule');
%

if nargin < 3, Options = []; end
if ~isfield( Options, 'maxDev'), Options.maxDev = 1e-6; end
if ~isfield( Options, 'verb'), Options.verb = 1; end

K        = kernel( Model.SV, Model.kernelName, Model.kernelArgs);
wNorm    = Model.Alpha(:)'*K*Model.Alpha(:);
tmpAlpha = Model.Alpha(:);
tmpSV    = Model.SV;

approxError = inf;
iter        = 0;
Z           =[];
Beta        = [];

while approxError > Options.maxDev & iter < newNumSV
  
  iter = iter + 1;

  if Options.verb, fprintf('Iteration %d: ', iter); end
  
  z = preimage_rbf( tmpSV, tmpAlpha, Model.kernelArgs );
  Z = [Z z];

  % TODO: the following computations can be speeded up significantly 
  %  by online updating the quantities
  Kz   = kernel( Z, Model.kernelName, Model.kernelArgs);
  Kzs  = kernel( Z, Model.SV, Model.kernelName, Model.kernelArgs);
  Beta = inv( Kz ) * Kzs * Model.Alpha(:);

  approxError = wNorm + Beta'*Kz*Beta - 2*Beta'*Kzs*Model.Alpha(:);
  
  if Options.verb, fprintf('|W - reducedW|^2 = %f\n', approxError ); end

  tmpAlpha = [Model.Alpha(:); -Beta(:)]';
  tmpSV    = [tmpSV z];
  
end

ReducedModel = Model;
ReducedModel.Alpha = Beta(:)';
ReducedModel.SV = Z;

return;
%EOF