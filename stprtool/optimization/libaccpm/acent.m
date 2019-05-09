function [x_star, H, niter] = acent(A,b,x_0) 
% ACENT computes the analytical center of the linear inequalities 
%       {Ax <= b}, i.e., the solution to 
%           minimize    - sum_{i=1,...,m}log(b_i - a_i^Tx) 
%       using the infeasible start Newton method, i.e. solving 
%           minimize    -sum_{i-1,...,m}log(y_i) 
%               s.t.    y = b - Ax 
%       where we initialize the procedure with y_0 > 0. 
%       The initial point x_0 need not be strictly inside the 
%       polyhedron {x | Ax <= b}. 
%


MAXITERS = 5000;
ALPHA = 0.1;
BETA  = 0.5;
TOL = 1e-6;
cond_thre = 1e14;

%
% INFEASIBLE NEWTON 
%

m = length(b);
n = length(x_0);

% starting point 
e = 0.01;
y = pos(A*x_0 - b + e); 
y(find(y==0)) = 1;
x = x_0; 
w = zeros(m,1);                 % dual variable

for iters = 1:MAXITERS 
    g = -1./y;
    H = diag(1./(y.^2)); 
    rd = g + w; 
    rp = A*x + y - b;
    res = [A'*w;rd;rp];
    
    if (norm(res) < sqrt(TOL)), break; end;
    
%     coef = 10;
%     
%     H = H + coef*A'*eye(size(H,1))*A;
%     g = g + coef*A'*eye(size(H,1))*h;
    
%     coef = 1;
%     nH = H;
%     ng = g;
%     
%     while true
%         
%         if cond(A'*nH*A) > 1e10
%             nH = H + coef*A'*eye(size(H,1))*A;
%             ng = g + coef*A'*eye(size(H,1))*h;
%             coef = coef * 10;
%         else 
%             H = nH;
%             g = ng;
%             break;
%         end    
%     end
    
%     AA = A'*H*A;
% 
%     if cond(AA) > cond_thre
%     
%         S = diag( 1./max(AA(:)) * ones(1,size(AA,1)) );
%         AS = AA*S;
%         dx = AS \ (A'*(g - H*rp));
%         dx = diag( max(AA(:)) * ones(1,size(AA,1)) ) * dx;
%         
% %         fprintf('bad condition number iter = %d\n', iters);
%        
% %         [R,p] = chol(AA);
% %         dx = R \ R'\(A'*(g - H*rp));
%         
% %        save('acent.mat', 'A','b','x_0');
%        
% %        error('bad cond number of matrix');
%         
% %        x_star = [];
% %        H = [];
% %        niter = [];
% %        return;
%     else
%         dx = (A'*H*A)\(A'*(g - H*rp));        
%     end
    
    dx = (A'*H*A)\(A'*(g - H*rp));        
    dy = - A*dx - rp;
    dw = - H*dy - rd; 

    t = 1;
    
    while(min(y+t*dy) <= 0) 
        t = BETA*t;
    end
    
    newres = [A'*(w+t*dw); w + t*dw - 1./(y+t*dy); ...
              A*(x+t*dx) + y + t*dy - b];
          
    while (norm(newres) > (1-ALPHA*t)*norm(res)) 
        t = BETA*t; 
        newres = [A'*(w+t*dw); w + t*dw - 1./(y+t*dy); ...
                  A*(x+t*dx) + y + t*dy - b];
    end
    
    x = x + t*dx; y = y + t*dy; w = w + t*dw;
end

%if iters >= MAXITERS
%    x_star = [];
%    H = [];
%    niter = [];
%    return;
%end

x_star = x; 
H = A'*diag((b-A*x).^(-2))*A;
niter = iters - 1;              % total number of newton steps taken


function y = pos( x )

% POS    Positive part.
%    POS(X) = MAX(X,0). X must be real.
%
%    Disciplined convex programming information:
%        POS(X) is convex and nondecreasing in X. Thus when used in CVX
%        expressions, X must be convex (or affine).

if nargin ~= 1, error('one input argument is expected.'); end
if ~isreal( x ),

	error( 'Argument must be real.' );
	
else

	y = max( x, 0 );

end

% Copyright 2010 Michael C. Grant and Stephen P. Boyd. 
% See the file COPYING.txt for full copyright information.
% The command 'cvx_where' will show where this file is located.