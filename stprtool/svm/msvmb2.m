function model = msvmb2( X, Y, C, kernelName, kernelArgs, options )
% MSVMB2 Multi-class SVM with L2-soft margin and L2-regularized bias.
%
% Synopsis:
%  model = msvmb2( X, Y, C, kernelName, kernelArgs )
%  model = msvmb2( X, Y, C, kernelName, kernelArgs, options )
%
% Description:
%
%  For more info refer to V.Franc: Optimization Algorithms for Kernel 
%  Methods. Research report. CTU-CMP-2005-22. CTU FEL Prague. 2005.
%  ftp://cmp.felk.cvut.cz/pub/cmp/articles/franc/Franc-PhD.pdf .
%
% Input:
%   X [...] Training inputs.
%   Y [nExamples x 1] Labels (1,2,...,nY).
%   C [1x1] Regularization constant.
%   kernelName [string] Kernel identifier. See HELP KERNEL.
%   kernelArgs [...] Kernel arguments.
%
%  options [struct] Control parameters:
%   .maxIter [1x1] Maximal number of iterations (default inf).
%   .tolAbs [1x1] Absolute tolerance stopping condition (default 0.0).
%   .tolRel [1x1] Relative tolerance stopping condition (default 0.001).
%   .qpBound [1x1] Thereshold on the primal objective (default 0).
%   .bufSize [1x1] # of columns of kernel matrix to be cached (default 1000).
%   .sv [string] Defines how to treat support vectors.
%     'nonzero'    model.SV = X(:,sum(abs(Alpha)) > 0)
%     'all'        model.SV = X
%     'none'       model.SV = []
%
%   .verb [1x1] If > 0 then some info is printed (default 0).
%
% Output:
%  model [struct] Multi-class SVM classifier:
%   .Alpha [nY x nSV] Dual weights.
%   .bias [nY x 1] Biases.
%   .SV [...] Support vectors (see option.sv).
%   .kernelName [string] 
%   .kernelArgs [...]
%   .options [struct] Copy of input options.
%   .stat [struct] Statistics about optimization:
%     .exitflag [1x1] Exitflag of the QP solver:
%       0  ... Maximal number of iterations reached: nIter >= MaxIter.
%       1  ... Relative tolerance reached: QP-QD <= abs(QP)*TolRel
%       2  ... Absolute tolerance reached: QP-QD <= TolAbs
%       3  ... Objective value reached threshold: QP <= QP_TH.
%     .QP [1x1] Primal value.
%     .QD [1x1] Dual value.
%     .nIter [1x1] Number of iterations of the QP solver.
%     .nKerEvals [1x1] Number of kernel evaluations.
%     .KerCols [1x1] Number of requested columns of virtual kernel matrix.
%     .cputime [1x1] Total time spent in SVM solver.
%
% Example:
%  data = load('fiveclassproblem');
%  kernelName = 'rbf'; kernelArgs= 0.5; C=10;
%  Model = msvmb2( data.X, data.Y, C, kernelName, kernelArgs )
%  figure; 
%  ppatterns(data.X,data.Y); 
%  ppatterns(Model.SV,[],'style','Encircle');
%  pclassifier(Model);
%
% See also 
%

tic;

%% process inputs 
if nargin < 5 || nargin > 6
    error('Incorrect number of input arguments.');
end
if nargin ~= 6 
    options = [];
end

%% default settings
if ~isfield(options,'maxIter'), options.maxIter = inf; end
if ~isfield(options,'tolAbs'), options.tolAbs = 0; end
if ~isfield(options,'tolRel'), options.tolRel = 0.001; end
if ~isfield(options,'qpBound'), options.qpBound = 0; end
if ~isfield(options,'bufSize'), options.bufSize = 1000; end
if ~isfield(options,'sv'), options.sv = 'nonzero'; end
if ~isfield(options,'verb'), options.verb=0; end

[nDim,nExamples] = size(X);
nY = max(Y);

%% call MEX implementation
[Alpha,bias,exitflag,nKerEvals,nKerCols,nIter,QP,QD] = ...
        msvmb2_mex(X,Y,C,kernelName,kernelArgs,...
            options.maxIter,...
            options.tolAbs, ...
            options.tolRel,...
            options.qpBound,...
            options.bufSize, ...
            options.verb );


%% get statistics
Stat.exitflag = exitflag;
Stat.QP=QP;
Stat.QD=QD;
Stat.nIter=nIter;
Stat.nKerEvals = nKerEvals;
Stat.nKerCols = nKerCols;
Stat.cputime = toc;
        
%% setup model
switch options.sv
    case 'nonzero'
        svIdx = find( sum(abs(Alpha)) > 0 );
        model.Alpha = Alpha(:,svIdx);
        model.SV = kernel_get_data(X,svIdx,kernelName);
        
    case 'all'
        model.Alpha = Alpha;
        model.SV = X;
        
    case 'none'
        model.Alpha = Alpha;
        
    otherwise
        error('Unsupported value of option.sv. ');
end

model.bias = bias;
model.C = C;
model.kernelName = kernelName;
model.kernelArgs = kernelArgs;
model.Options = options;
model.Stat = Stat;
model.eval = @kernelclassif;

return;

% EOF
