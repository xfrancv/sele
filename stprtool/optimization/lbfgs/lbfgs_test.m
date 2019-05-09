% Compare FMINUNC and LBFGS solver on a benchmark function from 
%  original Nocedal's code. 
%

% initial solution
n = 100;  % num of parameters; must be >= 2
x0 = zeros(n,1);
for i = 1 : 2 : n
    x0(i)   = -10.2;
    x0(i+1) = 10.2;
end


%% use Matlab solver
Opt1 = optimset('gradobj', 'on', ...
               'LargeScale', 'off', ...
 			   'display', 'iter-detailed');

t0=cputime;
[x1, fval1, exitFlag1] = fminunc(@(p) bench_fun_lbfgs([],p), x0, Opt1);
runTime1 = cputime-t0;

%% use LBFGS
Opt2.maxIter = 100;
Opt2.eps     = 1.e-5;
Opt2.m       = 5;
Opt2.verb    = 1;

t0=cputime;
[x2, fval2, exitFlag2] = lbfgs([],'bench_fun_lbfgs', x0, Opt2 );
runTime2 = cputime-t0;

%% Compare both solvers
fprintf('\n\n\ncomparison:\n');
fprintf('fminunc: fval-fval_opt=%f  runtime=%f[s]  norm(x-x_opt)=%f\n', ...
    fval1, runTime1, norm(x1(:)-ones(n,1)));
fprintf('lbfgs  : fval-fval_opt=%f  runtime=%f[s]  norm(x-x_opt)=%f\n', ...
    fval2, runTime2, norm(x2(:)-ones(n,1)));
