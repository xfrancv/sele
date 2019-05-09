function nY = maptoequbins(Y,B)
% MAPTOEQUBINS
%
% Synopsis:
%   Y = maptoequbins( X, nBins )
%
% Description:
%   The function maps input natural numbers to numbers from 1 to nBins,
%   such that the number vector will have equalized histogram and
%   the map preserves order, i.e.
%      if X(i) >= X(j) then Y(i) >= Y(j)
%
% Examples:
%   X = [1:10];
%   Y = maptoequbins(X,2)
%
%   X = rand(1,100);
%   Y = maptoequbins(X,10);
%   figure; hist(Y,10);
%


n = length(Y);

r = mod(n,B);
div = floor((n-r)/B);

groups = ones(B,1) * div;
groups(1:r) = groups(1:r) + 1;
groups = cumsum(groups);
groups = groups(1:end-1) + 1;

s = zeros(n,1);
s(groups) = 1;
s = cumsum(s)+1;

[sY, idx] = sort(Y);

nY = 1:n;
nY(idx) = s;

end