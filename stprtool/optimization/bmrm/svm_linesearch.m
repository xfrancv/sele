function [xopt, fval] = svm_linesearch( A, B, C, D, E, F )
% SVM_LINESEARCH
%
%   [xopt, fval] = svm_linesearch( A, B, C, D, E, F )
%
% Description:
%    It solves
%       min_x   A*x^2 + B*x + sum max( C(i)*x + D(i), E(i)*x + F(i) )
%                            i=1:N
%    where N = size(C,1) and A >= 0.
%

    N = size(C,1);

    % 
    T = zeros(N,1);
    G = zeros(N,2);
    for i = 1 : N

        if C(i) ~= E(i)
            T(i)   = (F(i)-D(i))/(C(i)-E(i));        
        end

        G(i,1) = min( C(i), E(i));
        G(i,2) = max( C(i), E(i));    
    end

    [T,idx] = sort( T, 'ascend' );
    G       = G( idx, :);

    s = sum( G(:,1) );

    for i = 1 : N
        t  = T(i);

        u = 2*A*t + B + s;

        if u >= 0
            xopt = -(B + s)/(2*A);
            break;
        end

        s = s - G(i,1) + G(i,2);
        u = 2*A*t + B + s;
        if u >= 0
            xopt = t;
            break;
        end
    end
    if u < 0
        xopt = -(B + s)/(2*A);
    end

    fval = A*xopt^2 + B*xopt + sum( max( [C(:)'*xopt + D(:)' ; E(:)'*xopt + F(:)']));
end
