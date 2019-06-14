function [x, stat] = accpm( Data,risk,A,b,boxConstr,regLambda, Options)
%%function [x, stat] = ACCPM(data,risk,A,b,options,thresh, Lambda,boxConstrSize)
% ACCPM Analytic Center Cutting Plane Method.
%
% Synopsis:
% 
% [W, Stat] = ACCPM( Data,risk,A,b,boxConstr,lambda, Opt)
% [W, Stat] = ACCPM( Data,risk,[],[],boxConstr,lambda, Opt)
%  
% The function solves a convex problem
%    min lambda*W'*W + risk(Data, W)   
%     W
%
%   s.t.  A*W <= b  and  -boxConstr <= W <= boxConstr
%
% The value of lambda can be zero. The calling syntax of the risk function is
%
%     [ Fval, subgrad ] = risk( Data )
%     [ Fval, subgrad ] = risk( Data, W )
%
% which returns value and subgradient of the risk at W. Calling risk( Data ) 
% without the argument W assumes that W equals to all zero.
%
% Input:
%   Options [struct]
%     .tolRel  [1x1] Relative tolerance (default 1e-2).
%     .maxIter [1x1] Maximal number of iterations (defaul 1e5).
%     .verb    [1x1] if 1 print progress status (default 1)
%

% implemented by Kostia Antoniuk
% 2014-09-26 VF, small changes

if nargin < 7, Options = []; end
if ~isfield( Options, 'tolRel'),  Options.tolRel = 1e-2; end
if ~isfield( Options, 'maxIter'), Options.maxIter = 1e5; end
if ~isfield( Options, 'verb'),    Options.verb = 1; end

%if nargin < 6 || isempty(thresh), thresh = 1e-2; end;
%if nargin < 7 || isempty(Lambda), Lambda = 0; end;
%if ~isfield(options,'MaxIter'), Options.maxIter = 1e5; end
%if nargin < 8, boxConstr = 10; end
    
start_time = cputime;
time_step = [];
risk_time = 0;
tic;
%[f, g, data] = risk(data);
[f, g] = risk(Data);
risk_time = risk_time + toc;
time_step = [time_step, cputime - start_time];

% n = data.nY * data.nY * data.nG;
n = length(g);

%********************************************************************
% cutting plane method 
%********************************************************************
% number of iterations 
niter = Options.maxIter; 
% initial localization polyhedron {x | Cx <= d}
%R = 10; % OK for 3 classes
R=boxConstr;
% R = 100; 
C = [eye(n); -eye(n)];             
if length(boxConstr) == 1
    d = R*ones(2*n,1); 
else
    d = [R;R].*ones(2*n,1); 
end

% b = b - 0.001;

if ~isempty(A)
    C = [A; C];
    d = [b; d];
end

cnstr = size(A,1);
m = size(C,1);
N = 3*n; 

% initial point 
x = zeros(n,1); 
x_best = zeros(n,1);

f_save = []; f_best = [];
l_save = []; l_best = []; 

F_save = []; F_best = [];
L_save = []; L_best = []; 

tot_nt = [];                        % total number of newton steps per iter
algo_time = 0;

for iter = 1:niter 
          
    % fixing error
%    r = b - A*x;
 %   [r, idx] = min(r);
    
    if 0 %r < 0       
        idx = idx(1);
        
        g = A(idx,:);
        C = [g; C];
        d = [g*x; d];
        
        cnstr = cnstr + 1;
        m = m + 1;
        
    else
    
        f_save = [f_save f];
        f_best = [f_best min(f_save)];  
        if f_best(end) == f_save(end), x_best = x; end;
        
        F_save = [F_save f];
        F_best = [F_best min(F_save)]; 
        
        % update polyhedron 
        C = [C; g']; 
%         d = [d; g'*x];          s              % neutral cut
        d = [d; g'*x + f_best(end) - f];        % deep cut
        
    end
    
    % find analytical center of polyhedron
    [x, H, nt] = acent(C,d,x);        
    algo_time = algo_time + toc;
    
%    if isempty(x)
%        fprintf('trying chebyshev center instead of analytical\n');
%        x = chebyshev_center(C,d);
%    end
%     x = chebyshev_center(C,d);
%     nt = 1;
    tot_nt = [tot_nt nt];     
    % compute lower bound 
       
    if 1 %r >= 0
        
        df = (d - C*x);
        df(df<= eps) = eps;
        duals = 1./df;
%         duals(isinf(duals)) = 0;
        mu = duals(1:m);
        lambda = duals(m+1:end);
        lambda = lambda./sum(duals(m+1:end));
        mu = mu./sum(duals(m+1:end));
        lb = (f_best - d(m+1:end)')*lambda - d(1:m)'*mu; 
        l_save = [l_save lb];
        l_best = [l_best max(l_save)];     

        L_save = [L_save lb];
        L_best = [L_best max(L_save)];
        
        if Options.verb
            if mod(iter,Options.verb) == 0 || exitflag ~= -1
                fprintf('%4d: nt%4d Fs=%f, Fb=%f, Ls=%f, Lb =%f, Fb-Lb=%f, 1-Lb/Fb=%f, R=%f\n', ...
                    iter, nt, f_save(end), f_best(end), l_save(end), l_best(end), f_best(end) - l_best(end), (f_best(end) - l_best(end)) / f_best(end), f);
            end
        end
%         d - C*x
%         if f_best(end) - l_best(end) < f_best(end) * thresh, break; end;
        if (f_best(end) - l_best(end)) < Options.tolRel * f_best(end) || abs(f_best(end)) < Options.tolRel
            time_step = [time_step, cputime - start_time];
            break;
        end;
    
    end
    
    if length(d) > m + N -1 
        
        % ranking and dropping constraints 
        temp1 = C*x - d; 
        temp2 = m*sqrt(diag(C*(H\C'))); 

        temp1 = temp1(m+1:end);
        temp2 = temp2(m+1:end);

        r = temp1./temp2; 
        [~, ind] = sort(r, 1, 'descend'); 
        
        ind = ind(1:N-1);        
        ind = sort(ind);
        
        f_save = f_save(ind);
        f_best = f_best(ind);
        l_save = l_save(ind);
        l_best = l_best(ind);
        
        ind = m + ind;
        ind = [[1:m]';ind];
        
        C = C(ind,:); 
        d = d(ind);             
    end
   
    % evaluate function and subgradient at current x
    tic;
    [f, g] = risk(Data,x);
    f = regLambda * sum(x(:).*x(:)) / 2 + f;
    g = g + regLambda * x;
    risk_time = risk_time + toc;
    
    time_step = [time_step, cputime - start_time];
%     f = F'*x;
    
end

x = x_best;

stat = [];
stat.f_save = F_save;
stat.f_best = F_best;
stat.tot_nt = tot_nt;
stat.l_save = L_save;
stat.l_best = L_best;
stat.risk_time = risk_time;
stat.algo_time = algo_time;
stat.time_step = time_step;

end