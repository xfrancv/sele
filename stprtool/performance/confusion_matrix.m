function [C,map] = confusion_matrix( predY, trueY, nY )
% CONFUSION_MATRIX
%
% [C,map] = confusion_marix( predY, trueY )
% [C,map] = confusion_marix( predY, trueY, nY )
%

    
%     [newPredY, map] = maptonatural( predY );
%     newTrueY        = maplabels( trueY, map);

    if nargin < 3, nY = max(max( predY ),max( trueY)); end

    C = zeros(nY,nY);

    for y1 = 1 : nY
        for y2 = 1 : nY
            C(y1,y2) = sum( trueY(:) == y1 & predY(:) == y2);
        end
    end

end
