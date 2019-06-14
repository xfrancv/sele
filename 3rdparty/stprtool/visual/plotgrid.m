function h = plotgrid( nCols, nRows, lineStyle)
% PLOTGRID
%

    if nargin < 3, lineStyle = 'b-'; end

    h = [];
    for i = 1 : nCols + 1
        h = [h plot([i-0.5 i-0.5],[0.5 nRows+0.5],lineStyle)];
    end

    for j = 1 : nRows + 1
        h = [h plot([0.5 nCols+0.5],[j-0.5 j-0.5],lineStyle)];
    end
end