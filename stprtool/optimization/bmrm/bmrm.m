function [W,Stat,histW]= bmrm(data,risk,lambda,options)
% BMRM Bundle Method for regularized Risk Minimization.
%
% Synopsis: 
%  [W,Stat,histW] = bmrm( Data, risk, lambda)
%  [W,Stat,histW] = bmrm( Data, risk, lambda, Options )
%
% Description:
%  Bundle Method for regularized Risk Minimization (BMRM) is an algorithm 
%  for minimization of function
%
%    F(W) = 0.5*lambda*W'*W + risk( Data, W )
%
%  where R(data,W) is an arbitrary convex function of W. BMRM requires 
%  a function to evaluate R and its subgradient at W. This function is 
%   expected as the second input argument "risk". The calling syntax is
%
%    [ riskVal, subgrad ] = risk( Data)
%    [ riskVal, subgrad ] = risk( Data, W )
%
%  which returns function value and subgradient evaluated at W. 
%  Calling risk( Data) without the argument W assumes that W equals to all
%  zero (this is used by BMRM to get dimension of parameter vector). 
%
% Reference: 
%  Teo et al.: A Scalable Modular Convex Solver for Regularized Risk 
%  Minimization, KDD, 2007
%  
% Inputs:
%  Data   [anything] Data.
%  risk   [function] Risk function.
%  lambda [1x1]      Regularization parameter.
%
%  Options [struct] 
%   .tolRel  [1x1] Relative tolerance (default 1e-3). Halt optimization
%                  if Fp-Fd <= tolRel*Fp holds.
%   .tolAbs  [1x1] Absolute tolerance (default 0). Halt if Fp-Fd <= tolAbs
%                  where Fp = F(W) and Fd is a lower bound of F(W_optimal). 
%   .maxIter [1x1] Maximal number of iterations (default inf). Halt 
%                  optimization if nIter >= maxIter .
%   .verb    [1x1] if 1 print progress status (default 1).
%   .bufSize [1x1] Size of the CP buffer in the number of CPs. 
%                  If specified then bufSizeMB is ignored. 
% 
% Outputs:
%  W [nDim x 1] Solution vector.
%  Stat [struct] 
%   .Fp    [1x1] Primal objective value.
%   .Fd    [1x1] Reduced (dual) objective value.
%   .nIter [1x1] Number of iterations.
%
%  histW   [nDim x nIter] Solution at iteration.
%

%
% 2010-04-12, Vojtech Franc
    
t0 = cputime;

% default options
if nargin < 4, options = []; end
if ~isfield(options,'tolRel'),      options.tolRel = 1e-3; end
if ~isfield(options,'tolAbs'),      options.tolAbs = 0; end
if ~isfield(options,'maxIter'),     options.maxIter = inf; end
if ~isfield(options,'verb'),        options.verb = 1; end
if ~isfield(options,'bufSize'),     options.bufSize = 500; end
if ~isfield(options,'cleanBuffer'), options.cleanBuffer = inf; end

%% 
tmp_time    = cputime;
[R,subgrad] = risk(data);
risktime1   = cputime-tmp_time;

%% if subgradient at w=0 is also zero vector then the optimum solution 
% is w=0 and the optimal value is R(0).
if all(subgrad==0)
    W = zeros(size(subgrad));
    Stat = [];
    Stat.Fp = R;
    Stat.Fd = R;
    Stat.nIter = 0;
    Stat.hist.Fp = R;
    Stat.hist.Fd = R
    Stat.hist.R = R;
    Stat.hist.qptime = 0;
    Stat.hist.risktime = risktime1;
    Stat.hist.hessiantime = 0;
    Stat.hist.innerlooptime = 0;
    Stat.hist.wtime = 0;
    Stat.hist.runtime = cputime-t0;
    return;
end


%% get paramater space dimension
nDim = length(subgrad);

%% computes number of cutting planes from given mega bytes
%if ~isfield(options,'bufSize')
%    tmp = whos('subgrad');
%    nCutingPlanesToBuffer = round( options.bufSizeMB*1024^2/tmp.bytes );
%%    nCutPlanes = (-nDim*8 + sqrt((nDim*8)^2 +4*8*1024^2 * options.bufSizeMB))/16;
%%    options.bufSize = round(nCutPlanes);
%    options.bufSize = nCutingPlanesToBuffer;
%end

%% inital solution
W = zeros(nDim,1);

if isa(subgrad,'double')    
    if issparse(subgrad)
        A = sparse(nDim,options.bufSize);
    else
        A = zeros(nDim,options.bufSize);
    end
else
    A = zeros(nDim,options.bufSize,class(subgrad));
end
b = zeros(options.bufSize,1);
H = zeros(options.bufSize,options.bufSize);

A(:,1) = subgrad;
b(1) = R;
alpha = [];

nIter = 0;
nCP = 0;
exitflag= -1;

% alloc buffers for meassured statistics
hist_Fd = zeros(options.bufSize+1,1);
hist_Fp = zeros(options.bufSize+1,1);
hist_R = zeros(options.bufSize+1,1);
hist_runtime = zeros(options.bufSize+1,1);
hist_risktime = zeros(options.bufSize+1,1);
hist_qptime = zeros(options.bufSize+1,1);
hist_hessiantime = zeros(options.bufSize+1,1);
hist_innerlooptime = zeros(options.bufSize+1,1);
hist_wtime = zeros(options.bufSize+1,1);

hist_risktime(1) = risktime1;
hist_runtime(1) = cputime-t0;
hist_Fd(1) = -inf;
hist_Fp(1) = R+0.5*lambda*norm(W)^2;
hist_R(1) = R;

if options.verb
    tmp1 = whos('A');
    tmp2 = whos('H');
    fprintf('Buffers allocated for %d cutting planes:\n',options.bufSize);
    fprintf('Cutting plane buffer: %f MB\n',tmp1.bytes/1024^2);
    fprintf('Hessian: %f MB\n',tmp2.bytes/1024^2);
    
    fprintf('%4d: tim=%.3f, Fp=%f, Fd=%f, R=%f\n', ...
        nIter, hist_runtime(1), hist_Fp(1), hist_Fd(1), hist_R(1));
end

%
if nargout >= 3, storeW = true; else storeW = false; end

if storeW, histW = zeros(nDim,options.bufSize); end


%% main loop
while exitflag == -1
    iterstart_time = cputime;
    
    nIter = nIter + 1;
    
    nCP = nCP + 1;
    %    H = A(:,1:nIter)'*A(:,1:nIter)/lambda;
    tmp_time = cputime;
    if nCP > 1,
%            H(1:nCP-1,nCP) = full(A(:,1:nCP-1)'*A(:,nCP))/lambda;
        H(1:nCP-1,nCP) = mult_matrices( A(:,1:nCP-1)', A(:,nCP) )/lambda;
        H(nCP,1:nCP-1) = H(1:nCP-1,nCP)';
    end
    %    H(nCP,nCP) = full(A(:,nCP)'*A(:,nCP))/lambda;
    H(nCP,nCP) = mult_matrices( A(:,nCP)', A(:,nCP) )/lambda;
    hist_hessiantime(nIter+1) = cputime-tmp_time;
    
    % solve reduced problem
    tmp_time = cputime;
    if all(b(1:nCP)==0)
        % if b is zero vector then the optimum is also zero vector => QP
        % doesn't need to be called
        alpha = zeros(nCP,1);
        Stat.QP = 0;
    else
        [alpha,Stat] = libqp_splx(H(1:nCP,1:nCP),-b(1:nCP),1,ones(1,nCP),1,[alpha;0]);
    end
    hist_qptime(nIter+1) = cputime-tmp_time;

    tmp_time = cputime;
%    W = -A(:,1:nCP)*alpha/lambda;
    W = -mult_matrices( A(:,1:nCP), alpha )/lambda;
    hist_wtime(nIter+1) = cputime - tmp_time;

    nzA = sum(alpha > 0);

    tmp_time = cputime;
    [R,subgrad] = risk(data,W);
    hist_risktime(nIter+1) = cputime-tmp_time;
        
    A(:,nCP+1) = subgrad;
%    b(nCP+1) = R - A(:,nCP+1)'*W;
    b(nCP+1) = R - mult_matrices( A(:,nCP+1)', W );
    
    Fp = R+0.5*lambda*norm(W)^2;
    Fd = -Stat.QP;
                  
    if Fp-Fd<= options.tolRel*abs(Fp)
        exitflag= 1;
    elseif Fp-Fd <= options.tolAbs
        exitflag= 2;    
    elseif nIter >= options.maxIter
        exitflag= 0;
    end 
    
    hist_runtime(nIter+1) = cputime-t0;
    hist_Fp(nIter+1) = Fp;
    hist_Fd(nIter+1) = Fd;
    hist_R(nIter+1) = R;
        
    % clean buffer
    if mod(nIter,options.cleanBuffer) == 0 && nzA < nCP
        old_nCP = nCP;
        idx = find(alpha > 0);
        nCP = length(idx);

        alpha = alpha(idx);
        H(1:nCP,1:nCP) = H(idx,idx);
        A(:,1:nCP) = A(:,idx);        
        b(1:nCP) = b(idx);        
        
        if options.verb
            fprintf('Cutting plane buffer cleaned up (old_nCP=%d, new_nCP=%d) \n',old_nCP, nCP);
        end
    end
    
    hist_innerlooptime(nIter+1) = cputime-iterstart_time;

    if (mod(nIter,options.verb) == 0 || exitflag ~= -1) & options.verb ~= 0
        fprintf(['%4d: tim=%.3f, Fp=%f, Fd=%f, (Fp-Fd)=%f, (Fp-Fd)/Fp=%f,' ...
                 'R=%f, nCP=%d, nzA=%d, timinner=%f, timrisk=%f, timw=%f, timqp=%f, timhes=%f\n'], ...
                nIter, hist_runtime(nIter+1), Fp, Fd, Fp-Fd,(Fp-Fd)/Fp, R,...
                nCP, nzA,hist_innerlooptime(nIter+1),hist_risktime(nIter+1),...
                hist_wtime(nIter+1), hist_qptime(nIter+1),hist_hessiantime(nIter+1));
    end
    
    if storeW, histW(:,nIter) = W; end
end

if options.verb
    fprintf('Accumulated times\n');
    fprintf('risk time       : %f\n', sum(hist_risktime(1:nIter+1)));
    fprintf('qptime          : %f\n', sum(hist_qptime(1:nIter+1)));
    fprintf('hessian time    : %f\n', sum(hist_hessiantime(1:nIter+1)));
    fprintf('w time          : %f\n', sum(hist_wtime(1:nIter+1)));
    fprintf('inner loop time : %f\n', sum(hist_innerlooptime(1:nIter+1)));
    fprintf('total runtime   : %f \n', hist_runtime(nIter+1));
end

Stat = [];
Stat.Fp = Fp;
Stat.Fd = Fd;
Stat.nIter = nIter;
Stat.hist.Fp = hist_Fp(1:nIter+1);
Stat.hist.Fd = hist_Fd(1:nIter+1);
Stat.hist.R = hist_R(1:nIter+1);
Stat.hist.qptime = hist_qptime(1:nIter+1);
Stat.hist.risktime = hist_risktime(1:nIter+1);
Stat.hist.hessiantime = hist_hessiantime(1:nIter+1);
Stat.hist.innerlooptime = hist_innerlooptime(1:nIter+1);
Stat.hist.wtime = hist_wtime(1:nIter+1);
Stat.hist.runtime = hist_runtime(1:nIter+1);

if storeW, histW = histW(:,1:nIter); end

return;
