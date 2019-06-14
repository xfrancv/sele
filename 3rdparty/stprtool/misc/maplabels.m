function dstLabels = maplabels(srcLabels, map)
% MAPLABELS
%
% Synopsis:
%   dstLabels = maplabels( srcLabels, map )
%
% Description:
%   If MAP is a cell then it maps labels in srcLabels to dstLabels 
%   from [1:numel(map)] such that ismember( srcLabel(i), map{dstLabel(i)} ) 
%   is true.
%
%   If MAP is a vector then it substitues labels in map by 
%   labels [1:length(map)].
%
% Example:
%   maplabels( [1:10], {[1:5],[6:10]} ) 
%

    if iscell( map )
        dstLabels = zeros( size( srcLabels ) );
        for y = 1 : numel(map)
            Z = map{y};
            for z = Z(:)'
                idx            = find( srcLabels == z );
                dstLabels(idx) = y;
            end
        end
    else
        dstLabels = zeros( size( srcLabels ) );
        for y = 1 : numel(map)
            dstLabels( find( srcLabels == map(y))) = y;
        end        
    end
    
return;
