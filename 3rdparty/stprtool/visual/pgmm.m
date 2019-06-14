function varargout = pgmm( model, varargin )
% PGMM Vizualizes Gaussian Mixture Model.
%
% Synopsis:
%  pgmm( model );
%  pgmm( model, options );
%  pgmm( model, 'item1',value1,... );
%  h = pgmm( ... );
%
% Description:
%  Let p(x) be a Gaussian Mixture Model
%   p(x) =  sum p(x|y) * p(y)
%          y=1:nY
%  where p(x|y), y=1:nY are Gaussian components and p(y) is a discrete 
%  probability distribution. 
%
%  PGMM visualizes the Gaussian Mixture Model by plotting isolines
%  (contours) of the probability distribution functions p(x) (default) 
%  or p(x,y) using pgmm(model,'comp',1).
% 
% Input:
%  model [struct] Gaussiona Mixture Model (see HELP GMM_ML).
%
%  options.comp [1x1] If 1 then it plots mixture components (default 0).
%  options.adj_axes [1x1] If 1 then axes are set to display whole 
%    mixture otherwise unchanged (default 1).
%  options.clabels [1x1] If 1 then display contour labels (default 1).
%  options.values [1xM] Values defining the isolines.
%
% Output:
%  h [1 x nObjects] Handles of graphics object.
%
% Example:
%  !!! TODO: improve the example; to implement 1D variant !!!
%
%  load('riply_dataset','trn_X','trn_y');
%  model = gmm_ml( trn_X, trn_y );
%  figure; hold on; ppatterns(trn_X,trn_y); pgmm( model );
%  figure; hold on; ppatterns(trn_X,trn_y); pgmm( model,'comp',1 );
%

if nargin >= 2, options=c2s(varargin); else options = []; end
if ~isfield(options,'adj_axes'), options.adj_axes = 1; end
if ~isfield(options,'clabels'), options.clabels = 1; end
if ~isfield(options,'values'), options.values = []; end
if ~isfield(options,'linestyle'), options.linestyle = []; end
if ~isfield(options,'comp'), options.comp=0; end

GRID = 150;

[nDim,nY] = size( model.Mean );
a = axis;

if nDim == 1
    % 1d case
    min_x = a(1);
    max_x = a(2);
    if options.adj_axes == 1,   
        margin = sqrt(max(model.D(:)))*3;
        min_x = min(min_x,min(model.Mean)-margin);
        max_x = max(max_x,max(model.Mean)+margin);
    end

    Ax = linspace(min_x,max_x,GRID);
    Px = exp(gmm_logpx(Ax(:)', model ));
    if ~isempty(options.linestyle)
         h = plot(Ax,Px,options.linestyle);
    else
         h = plot(Ax,Px);
    end        
    
elseif nDim ==2,
    % 2d case
    min_x = a(1);
    max_x = a(2);
    min_y = a(3);
    max_y = a(4);

    if options.adj_axes == 1,   
        margin = sqrt(max(model.D(:)))*3;
        min_x = min(min_x,min(model.Mean(1,:))-margin);
        max_x = max(max_x,max(model.Mean(1,:))+margin);
        min_y = min(min_y,min(model.Mean(2,:))-margin);
        max_y = max(max_y,max(model.Mean(2,:))+margin);
    end
  
    [Ax,Ay] = meshgrid(linspace(min_x,max_x,GRID), linspace(min_y,max_y,GRID));
    
    if options.comp == 0
        Px = exp(gmm_logpx([Ax(:)';Ay(:)'], model ));
        args = [];
        if ~isempty(options.values)
            args{1} = options.values;
        end
        if ~isempty(options.linestyle)
            args{end+1} = options.linestyle;
        end
        if ~isempty(args)
            [c,h] = contour( Ax, Ay, reshape(Px,GRID,GRID),args{:}); 
        else
            [c,h] = contour( Ax, Ay, reshape(Px,GRID,GRID)); 
        end
        if options.clabels==1
            clabel(c,h);
        end
    else
        Pxy = exp(gmm_logpxy([Ax(:)';Ay(:)'], model ));
        h=[];
        for y=1:nY
            args = [];
            if ~isempty(options.values)
                args{1} = options.values;
            end
            if ~isempty(options.linestyle)
                args{end+1} = options.linestyle;
            end
            if ~isempty(args)
                [c,hh] = contour( Ax, Ay, reshape(Pxy(y,:),GRID,GRID),args{:}); 
            else
                [c,hh] = contour( Ax, Ay, reshape(Pxy(y,:),GRID,GRID)); 
            end
%            if ~isempty(options.values)
%                [c,hh] = contour( Ax, Ay, reshape(Pxy(y,:),GRID,GRID),options.values); 
%            else
%                [c,hh] = contour( Ax, Ay, reshape(Pxy(y,:),GRID,GRID)); 
%            end
            if options.clabels==1
                clabel(c,hh);
            end
            h=[h hh];
        end
    end

    for y=1:nY
        hh = plot(model.Mean(1,y),model.Mean(2,y),'+k');
        h = [h hh];
        set(hh,'Markersize',16);
        hh = text(model.Mean(1,y),model.Mean(2,y),num2str(y));
        h = [h hh];
        set(hh,'HorizontalAlignment','left');
        set(hh,'VerticalAlignment','bottom');
        set(hh,'Color','k');
        set(hh,'FontSize',13);
        
    end 
end

% return handles if required
if nargout > 0, varargout{1} = h; end

return;
