function logSumA = logsumexp(logA)
%LOGSUMEXP Computes logarithm of a sum of very small numbers.
%
%Synopis:
%   logSumA = logsumexp(logA)
%
%Description:
% This function computes
%
%   logSumA = log( sum( A ) )
%
% from logA = log(A), i.e. from logarithm of the summands where A [N x 1] 
% is a vector of positive reals numbers. 
%
% This function is equivalent to log(sum(exp( logA ))) . The function
% logsumexp is useful if A are extremaly small numbers in which case 
% log(sum(exp(logA))) would not work due to a finite precision of double.
%
% If A is a matrix N x M then the output is a matrix 1 x M whose 
% elements contain logSumA(i) = log( sum( A(:,i) ) ).
%
%Example:
% logA = [-1000 -1000 -1000 -1000 -1000 -1000 -1000 -1000]'
% log(sum(exp(logA)))
% logsumexp(logA)
%
% logA = [-100 -100 -100 -100 -100 -100 -100 -100]'
% log(sum(exp(logA)))
% logsumexp(logA)
%

%if min(size(logA)) == 1
%    sorted_logA = sort(logA(:));
%else
%end

    if size(logA,1) == 1
        logSumA = logA;
    else

        sorted_logA = sort(logA);    
        logSumA     = sorted_logA(1,:);

        for i=2:size(sorted_logA,1)
            % Patch sent by Lukas Cerman:
            % Old:
            % logSumA = sorted_logA(i,:) + log(1 + exp(logSumA - sorted_logA(i,:)));
            % New:
            sel = isfinite(sorted_logA(i,:));
            logSumA(sel) = sorted_logA(i,sel) + log(1 + exp(logSumA(1,sel) - sorted_logA(i,sel)));
        end
    end
end