function [X,Y] = cumhist( V )
% CUMHIST creates a cumulative histogram.
%
% Synopsis:
%    [X,Y] = cumhist( V )
%
% Description:
%   Y(i) is a ratio of the number of values in V not greater
%   then X(i) and the length of V.
%
    N = length( V );
    X = sort( V );
    Y = [1:N]/N;
end