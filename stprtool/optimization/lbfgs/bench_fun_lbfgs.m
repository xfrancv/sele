function [fval,grad] = bench_fun_lbfgs(Params,x)
% BENCH_FUN_LBFGSL Benchmark function from Nocedal's code implementing LBFGS.
%
% Synopsis:
%   [fval,grad] = bench_fun_lbfgsl(Params,x)
%
% The first argument "Params" is not used inside the function
%
% Description:
%   The optimal value is zero. The optimal point is all ones.
% 
    
    n = length( x );

    fval = 0;
    grad = zeros(n,1);
    
    for j = 2:2:n
      t1 = 1 - x(j - 1);
      t2 = (x(j) - x(j - 1)^2) * 10;
      
      grad(j) = t2 * 20;
      grad(j - 1) = (x(j - 1) * grad(j) + t1) * -2;
      
      fval = fval + t1*t1 + t2*t2;      
    end
    
end