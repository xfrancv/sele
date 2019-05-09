function s = sign2( x )
% SIGN2 returns 1 for non-negative numbers and -1 otherwise.
%
% Synopsis:
%   s = sign2( x )
%
% Example:
%  sign2([-10 0 10]) % returns [-1 1 1]
%

    s = 2*double( x >=0 ) - 1;

end

