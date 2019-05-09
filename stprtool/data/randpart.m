function [varargout]=randpart(N,portion)
% RANDPART Creates random partitions of a set 
%
% Synopsis:
%  [set1,set2,...] = randpart(N,portion)
%
% Input:
%  N [1x1] number of elements.
%  portion [Sx1] portion of elements the individual (sub)-set.
%
% Output:
%  set1 [1 x N1] elemets in the set1; N1 = round(N*portion(1)).
%  set2 [1 x N2] elemets in the set2; N2 = round(N*portion(2)).
%  ...
%
% Example:
%  rng(0);
%  [trnIdx,valIdx,tstIdx] = randpart( 100,[0.6 0.2 0.2])
%

    % 
    idx = randperm( N );
    
    begIdx = 1;
    for i = 1 : numel( portion)
        endIdx       = min(N, begIdx - 1 + round(portion(i)*N) );
        varargout{i} = idx(begIdx:endIdx);
        begIdx       = endIdx+1;    
    end
end
% EOF
