function h=plotbox(bBoxes,plotArgs,legend,textArgs,hAxis)
% PLOTBOX Plots box(es) to the current figure.
% 
% Synopsis:
%  h=plotbox(box)
%  h=plotbox(box,plotArgs)
%  h=plotbox(bBoxes,plotArgs,legend,textArgs)
%
% Input:
%  box [4 x N] axis aligned box 
%    [top_left_col top_left_row bottom_right_col bottom_right_row]
%
%  box [8 x N] box in a general position
%    [x1 y1 x2 y2 x3 y3 x4 y4]
%
%  plotArgs [cell array] arguments passed to PLOT function.
%  legend [cell array of strings] text displayed below box.
%  textArgs [cell array] arguments past to TEXT function.
%
% Output:
%  h [s] handles of used graphical objects.
%
% Example:
%  figure; axis([-1 1 -1 1]);
%  plotbox([-0.5 -0.5 0.5 0.5]');
%
%  figure; axis([-1 1 -1 1]);
%  plotbox([-0.5 0 0 0.5 0.5 0 0 -0.5]');
%
 
    if nargin < 2 | isempty( plotArgs), plotArgs = {'b'}; end
    if nargin < 4 | isempty( textArgs), textArgs = {'color','b'}; end
    if nargin < 5, hAxis = []; end

    if ~iscell( plotArgs ), plotArgs = {plotArgs}; end
    
    isHold = ishold;
    hold on;

    [nPoints,nBoxes]  = size( bBoxes );
    h = [];

    for i = 1 : nBoxes

        box = bBoxes(:,i);

        if nPoints == 4
            
            if isempty( hAxis )
                h(end+1) = plot( box([1 3 3 1 1]),box([2 2 4 4 2]), plotArgs{:} );
            else
                h(end+1) = plot( hAxis, box([1 3 3 1 1]),box([2 2 4 4 2]), plotArgs{:} );
            end
                

           if nargin >= 3 & numel( legend ) >= i
               fs = min( 18, (box(3)-box(1)+1)/4);
               h(end+1) = text( box(1), box(4), legend{i}, 'verticalAlignment','top','fontsize',fs, textArgs{:} );
           end

        elseif nPoints == 8

            if isempty( hAxis )
                h(end+1) = plot( box([1 3 5 7 1 ]), box([2 4 6 8 2]), plotArgs{:} );
            else
                h(end+1) = plot( hAxis, box([1 3 5 7 1 ]), box([2 4 6 8 2]), plotArgs{:} );
            end
           
            if nargin >= 3 & numel( legend ) >= i
               fs = min( 18, sqrt((box(1)-box(3))^2+(box(2)-box(3))^2) /4);
               h(end+1) = text( min(box([1:2:end])), max(box([2:2:end])), legend{i}, 'verticalAlignment','top','fontsize',fs, textArgs{:} );
           end

        end
    end

    if ~isHold, hold off; end

%EOF


        
    
    