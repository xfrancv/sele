function [outputNumbers, map] = maptonatural(inputNumbers)
% MAPTONATURAL Map input sequence to natural numbers.
% 
% Synopsis:
%   [outputNumbers, map] = maptonatural(inputNumbers)
%
% Description:
%   The function assigns each unique number of A an index from
%   1 to N where N is the number of unique values in A.
%   For example, given a sequence
%     A = [1.3 -10 2.4 2.4 7]
%   calling
%    [B, map] = maptonatural(A)
%   produces
%     B = [2 1 2 2 3]
%   and map(A(i)) = B(i) for all i=1:length(A).
%

map = unique(inputNumbers);
for i=1:length(map)
    outputNumbers(find(map(i) == inputNumbers)) = i;
end

return;
% 