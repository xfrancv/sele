function Model = svmocas_kernel( X, X0, Y, C, kernelName, kernelArgs, Opt )
% SVMOCAS_KERNEL
%
% Synopsis:
%  Model = svmocas_kernel( X, X0, Y, C, kernelName, kernelArgs, Options )
%
% Description:
%  
% Input:
%   X [...] Training examples.
%   X0 [1x1] Constant feature added to all examples.
%   Y [numExamples x 1] Labels (+1 / -1).
%   C [1x1] Positive regularization constant.
%   kernelName [string] Kernel name. See HELP KERNEL.
%   kernelArgs [...] Kernel arguments. See HELP KERNEL.
%   Options [struct] Options for SVMOCAS. See HELP SVMOCAS.
%   
% Output:
%
% Example:
%
%  load('riply_dataset','trn','tst');
%  Model = svmocas_kernel(trn.X,1,trn.Y,10,'rbf',0.5);
%  trnErr = sum(kernelclassif( trn.X, Model ) ~= trn.Y(:)')/length(trn.Y)
%  tstErr = sum(kernelclassif( tst.X, Model ) ~= tst.Y(:)')/length(tst.Y)
%  figure;
%  ppatterns(trn.X,trn.Y);
%  pclassifier(Model);
%

if nargin < 6
    error('At least six arguments must be passed.');
end

if nargin < 7
    Opt = [];
end
if ~isfield(Opt,'method'), Opt.method = 1; end
if ~isfield(Opt,'tolRel'), Opt.tolRel = 0.01; end
if ~isfield(Opt,'tolAbs'), Opt.tolAbs = 0.01; end
if ~isfield(Opt,'qpBound'), Opt.qpBound = 0.01; end
if ~isfield(Opt,'bufSize'), Opt.bufSize = 1000; end
if ~isfield(Opt,'nExamples'), Opt.nExamples = inf; end
if ~isfield(Opt,'maxTime'), Opt.maxTime = inf; end

%% compute ortonormal bases for given RKHS
K = kernel( X, kernelName, kernelArgs );

[U,D] = svd(K);
A=diag(1./sqrt(diag(D)))*U';

linX = A*K;  % X contains linearized features

%% compute linear SVM classifier
[W,W0,Stat] = svmocas(linX,X0,Y,C,Opt.method,Opt.tolRel,Opt.tolAbs,Opt.qpBound,Opt.bufSize,Opt.nExamples,Opt.maxTime);

%% setup model
Model.Alpha = W'*A;
Model.Alpha = Model.Alpha(:)';
Model.bias = W0;
Model.C = C;
Model.SV = X;
Model.Options = Opt;
Model.Stat = Stat;
Model.kernelName = kernelName;
Model.kernelArgs = kernelArgs;
Model.eval  = @kernelclassif;

return;
