function H = ppatterns(X,Y,varargin)
% PPATTERNS Visualizes patterns (X,Y) as points in 2D.
% 
% Synopsis:
%  ppatterns(X)
%  ppatterns(X,Y)
%  ppatterns(X,[])
%  ppatterns(X,Y,'option1','value1','option2','value2'...)
%
% Input:
%  X [2 x nPoints] Feature vectors
%  Y [nPoints x 1] Discrete labels
%
%  Options:
%   'style' with possible values'B&W','MARKERS','CIRCLES',
%           'ENCIRCLE','ENSQUARE','INDEX', 'LABELS','WHITELABELS',
%           'DETECTION'
%
%   'lineWidth' 
%   'fontSize'
%   'colors'    colormap [M x 3] or a string ['rgb...']' compatibple with plot.
%   'markers'
%   'markerSize' 
%   'viewAz'
%   'viewEl'
%
% Output:
%  H [struct] Handles to graphical objects.
%
% Example:
%  load('riply_dataset','Trn'); 
%  switch input('select dimension from 1,2,3: ')
%     case 1, X = Trn.X(1,:);                      
%     case 2, X = Trn.X;                           
%     case 3, X = [Trn.X; Trn.X(1,:).*Trn.X(2,:)]; 
%  end
%  Y = Trn.Y;
%
%  figure; ppatterns(X);
%  figure; ppatterns(X,Y);
%  figure; ppatterns(X,Y,'style','circles');
%  figure; ppatterns(X,Y,'style','dots');
%  figure; ppatterns(X); ppatterns(X,Y,'style','encircle');
%  figure; ppatterns(X); ppatterns(X,Y,'style','ensquare');
%  figure; ppatterns(X,Y,'style','Index');
%  figure; ppatterns(X,Y,'style','labels');
%  

    holdIsOn = ishold;
    hold on;

    [nDim,nPoints] = size(X);
    if nDim > 3 || nDim < 1 || ~isa(X,'double')
        error('The first argument must be double matrix with 1,2, or 3 rows.'); 
    end


    %% default labels
    if nargin < 2 || isempty( Y )
        Y = zeros( nPoints, 1 );
    end
    nY = length( unique( Y ));


    %% default options
    if nargin >= 3, Opt = c2s( varargin ); else Opt = []; end

    if ~isfield(Opt,'markers')
        Opt.markers = '+o*spdv^<>ox'; 
    end
    if ~isfield(Opt,'colors')
        %Opt.colors  = ['brgcmybrgcrk']'; 
        Opt.colors  = [ [1 0 0]; ...% red
                        [0 0.5 0]; ...% greeen
                        [0 0 1]; ...% blue
                        [0 210 255]/255; ... % dark cyan
                        [255 0 255]/255; ... % pink
                        [124 255 0]/255; ... % light green
                        [255 252 0]/255; ... % yellow
                        [255 156 0]/256; ... % orange
                        [198 0 255]/255; ... % vinova
                        [169 56 0]/255;  ... % brown
                        [0 0 1];         ... % red
                        [0 0 0];         ... % black
                        ];
    end
    if ~isfield(Opt,'style'),      Opt.style   = 'markers'; end
    if ~isfield(Opt,'markerSize'), 
        if ~strcmpi(Opt.style,'dots')
            Opt.markerSize = 14; 
        else
            Opt.markerSize = 4; 
        end
    end
    if ~isfield(Opt,'lineWidth'),  Opt.lineWidth = 1; end
    if ~isfield(Opt,'fontSize'),   Opt.fontSize = 12; end
    if ~isfield(Opt,'viewAz'),     Opt.viewAz = -45; end
    if ~isfield(Opt,'viewEl'),     Opt.viewEl = 45; end


    %% enlarge axis to accomodate all points
    curAxis      = axis;
    switch size( X, 1)

        case 1
            desiredAxis = [min(X) max(X) curAxis(3:4) ];            
        case 2
            desiredAxis = [min(X(1,:)) max(X(1,:)) min(X(2,:)) max(X(2,:)) ];
        case 3
            desiredAxis = [min(X(1,:)) max(X(1,:)) min(X(2,:)) max(X(2,:)) min(X(3,:)) max(X(3,:))];
            if length( curAxis ) == 4, curAxis = [curAxis 0 0]; end
    end
    
    if ~is_contained( desiredAxis, curAxis )
        newAxis = bounding_box( desiredAxis, curAxis);
        axis( newAxis );
    end

    
    %%
    H.axis   = gca;
    H.fig    = gcf;
    H.points = [];
    H.colors = Opt.colors;
    H.style  = Opt.style;

    %%
    switch upper( Opt.style)

        case 'B&W'
          for y = min(Y):max(Y)

            idx = find( Y == y );        
            if ~isempty(idx)
%                H.points = [H.points plot( X(1,idx), X(2,idx), [getMarker(y,Opt) 'k'])];
                H.points = [H.points my_plot( X(:,idx), [getMarker(y,Opt) 'k'])];
            end
          end

        case 'DETECTION'
          
            idx1 = find( Y == 1 ); % pos class
            idx2 = find( Y ~= 1 ); % neg class
            
            if ~isempty(idx1)
                H.points = [H.points my_plot(X(:,idx1),...
                    'markersize', 4,...
                    'marker','+',...
                    'color', [1 0 0],...
                    'markerfacecolor', [1 0 0],...
                    'markeredgecolor', [1 0 0],...
                    'linestyle','none') ];
            end
            if ~isempty(idx2)
                H.points = [H.points my_plot(X(:,idx2),...
                    'markersize', 4,...
                    'marker','o',...
                    'color', [0 1 0],...
                    'markerfacecolor', [0 1 0],...
                    'markeredgecolor', [0 1 0],...
                    'linestyle','none') ];
            end
            
            
        case 'CIRCLES'

          for y = min(Y):max(Y)

            idx = find(Y == y);
            if ~isempty(idx)
                H.points = [H.points my_plot(X(:,idx),...
                    'markersize', Opt.markerSize,...
                    'marker','o',...
                    'color', getColor( y, Opt ),...
                    'markerfacecolor', getColor(y, Opt),...
                    'markeredgecolor', 'k',...
                    'linestyle','none') ];
%                 H.points = [H.points plot(X(1,idx),X(2,idx),...
%                     'markersize', Opt.markerSize,...
%                     'marker','o',...
%                     'color', getColor( y, Opt ),...
%                     'markerfacecolor', getColor(y, Opt),...
%                     'markeredgecolor', 'k',...
%                     'linestyle','none') ];
            end
          end
          
        case 'DOTS'

          for y = min(Y):max(Y)

            idx = find(Y == y);
            if ~isempty(idx)
                H.points = [H.points my_plot(X(:,idx),...
                    'markersize', Opt.markerSize,...
                    'marker','o',...
                    'color', getColor( y, Opt ),...
                    'markerfacecolor', getColor(y, Opt),...
                    'markeredgecolor', getColor(y, Opt),...
                    'linestyle','none') ];
%                 H.points = [H.points plot(X(1,idx),X(2,idx),...
%                     'markersize', 4,...
%                     'marker','o',...
%                     'color', getColor( y, Opt ),...
%                     'markerfacecolor', getColor(y, Opt),...
%                     'markeredgecolor', getColor(y, Opt),...
%                     'linestyle','none') ];
            end
          end

        case 'MARKERS'

          for y = min(Y):max(Y)

            idx = find(y == Y);
            if ~isempty(idx)
                H.points = [H.points my_plot(X(:,idx),...
                    'marker', getMarker( y, Opt ),...
                    'color', getColor( y, Opt ),...
                    'linestyle','none')];
%                 H.points = [H.points plot(X(1,idx),X(2,idx),...
%                     'marker', getMarker( y, Opt ),...
%                     'color', getColor( y, Opt ),...
%                     'linestyle','none')];
            end
          end
          
        case 'GRAY'

          Opt.colors = gray(round(1.1*nY));                        
          for y = min(Y):max(Y)
              
            idx = find(y == Y);
            if ~isempty(idx)
                H.points = [H.points my_plot(X(:,idx),...
                     'markersize', 10,...
                     'marker','o',...
                     'linewidth', Opt.lineWidth,...
                     'markerfacecolor', getColor(y, Opt),...
                     'markeredgecolor', 'none',...
                     'linestyle','none')];
%                 H.points = [H.points plot(X(1,idx),X(2,idx),...
%                      'markersize', 10,...
%                      'marker','o',...
%                      'linewidth', Opt.lineWidth,...
%                      'markerfacecolor', getColor(y, Opt),...
%                      'markeredgecolor', 'none',...
%                      'linestyle','none')];
            end
          end

        case 'ENCIRCLE'        

            for y = min(Y):max(Y),

                idx = find(y == Y);
                if ~isempty(idx)
                    H.points = [H.points my_plot(X(:,idx),...
                        'markersize', Opt.markerSize,...
                        'marker','o',...
                        'linewidth', Opt.lineWidth,...
                        'markerfacecolor','none',...
                        'markeredgecolor', getColor( y, Opt ),...
                        'linestyle','none')];
%                     H.points = [H.points plot(X(1,idx),X(2,idx),...
%                         'markersize', Opt.markerSize,...
%                         'marker','o',...
%                         'linewidth', Opt.lineWidth,...
%                         'markerfacecolor','none',...
%                         'markeredgecolor', getColor( y, Opt ),...
%                         'linestyle','none')];
                end
            end

        case 'ENSQUARE'

              for y = min(Y):max(Y)

                idx = find(y == Y);
                if ~isempty(idx)
                    H.points = [H.points my_plot(X(:,idx),...
                        'markersize', Opt.markerSize,...
                        'marker','s',...
                        'linewidth', Opt.lineWidth,...
                        'markerfacecolor','none',...
                        'markeredgecolor', getColor( y, Opt),...
                        'linestyle','none')];
%                     H.points = [H.points plot(X(1,idx),X(2,idx),...
%                         'markersize', Opt.markerSize,...
%                         'marker','s',...
%                         'linewidth', Opt.lineWidth,...
%                         'markerfacecolor','none',...
%                         'markeredgecolor', getColor( y, Opt),...
%                         'linestyle','none')];
                end
              end

        case 'INDEX'

          H.text = [];
          for y = min(Y):max(Y)

            idx = find(y == Y);
            if ~isempty(idx)
                H.points = [H.points my_plot(X(:,idx),...
                    'markersize', Opt.markerSize,...
                    'marker','o',...
                    'markerfacecolor','none',...
                    'markeredgecolor', getColor( y, Opt),...
                    'linestyle','none')];
%                 H.points = [H.points plot(X(1,idx),X(2,idx),...
%                     'markersize', Opt.markerSize,...
%                     'marker','o',...
%                     'markerfacecolor','none',...
%                     'markeredgecolor', getColor( y, Opt),...
%                     'linestyle','none')];

                for j = idx(:)'

                    h = my_text( X(:,j),num2str(j) );
%                     h = text( X(1,j), X(2,j),num2str(j) );
                    set(h,'HorizontalAlignment','center');
                    set(h,'VerticalAlignment','middle');
                    set(h,'Color','k' );
                    set(h,'FontSize', Opt.fontSize );
                    H.text = [H.text h]; 
                end

            end
          end

        case 'LABELS'

          H.text = [];
          for y = min(Y):max(Y)

            idx = find(y == Y);
            if ~isempty(idx)
                for j = idx(:)'

%                     h = text( X(1,j),X(2,j), num2str(y));
                    h = my_text( X(:,j), num2str(y));
                    set(h,'HorizontalAlignment','center');
                    set(h,'VerticalAlignment','middle');
                    set(h,'Color','k');
                    set(h,'FontSize', Opt.fontSize );
                    H.text = [H.text ; h(:)]; 
                end

            end
          end

        case 'WHITELABELS'

          H.text = [];
          for y = min(Y):max(Y)

            idx = find(y == Y);
            if ~isempty(idx)
                for j = idx(:)'
                    h = my_text( X(:,j), num2str(y));
%                     h = text( X(1,j), X(2,j), num2str(y));
                    set(h,'HorizontalAlignment','center');
                    set(h,'VerticalAlignment','middle');
                    set(h,'Color','w');
                    set(h,'FontSize', Opt.fontSize );
                    H.text = [H.text ; h(:)]; 
                end

            end
          end

        otherwise
            error('Unknown STYLE.');

    end

    if size(X,1) == 3,
        view( Opt.viewAz, Opt.viewEl );
    end
    if ~holdIsOn, hold off; end

end


function col = getColor( in, Opt)
    
    M   = size( Opt.colors, 1);
    idx = mod(in, M); if idx == 0, idx = M; end    
    col = Opt.colors( idx, : );

end

function col = getMarker( in, Opt)
    
    M   = length( Opt.markers );
    idx = mod(in, M); if idx == 0, idx = M; end    
    col = Opt.markers( idx );

end

% return true if innerBox [minx maxx miny maxy minz maxz] is insider 
% the outerBox.
function res = is_contained( innerBox, outerBox )

    innerBox = reshape( innerBox, 2, length( innerBox )/2);
    outerBox = reshape( outerBox, 2, length( outerBox )/2);

    res = all( innerBox(1,:) >= outerBox(1,:) & innerBox(2,:) <= outerBox(2,:));
end

% return minimal sized box containing box1 and box2
function newBox = bounding_box( box1, box2 )

    box1 = reshape( box1, 2, length( box1 )/2);
    box2 = reshape( box2, 2, length( box2 )/2);
    
    newBox = [min( box1(1,:), box2(1,:)) ; max( box1(2,:), box2(2,:))];
    newBox = newBox(:);
end

%
function res = my_plot( X, varargin )

    switch size(X, 1)
        case 1
            res = plot( X, zeros(1,length(X)), varargin{:} );
        case 2
            res = plot( X(1,:), X(2,:), varargin{:} );
        case 3
            res = plot3( X(1,:), X(2,:), X(3,:), varargin{:} );            
    end

end

%
function res = my_text( X, str)

    switch size(X, 1)
        case 1
            res = text( X, 0, str);
        case 2
            res = text( X(1), X(2), str);
        case 3
            res = text( X(1), X(2), X(3), str);
    end    
end

% EOF