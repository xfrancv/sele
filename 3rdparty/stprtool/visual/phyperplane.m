function h = phyperplane(w,w0,varargin)
% PHYPERPLANE visualize hyperplane in 3D.
%
%  h = phyperplane(w,w0)
%  h = phyperplane(w,w0,'option1','value1','option2','value2',...)
%
%  The hyperplane is a set of points x which satisfy 
%     x'*w + w0 = 0
% 
% Input:
%  w  [3 x 1] normal vector.
%  w0 [1 x 1] bias.
%  
%  options:
%   'viewAz' view-direction azimut (default -45)
%   'viewEl' view-direction elevation (default 45)
%   'color'  color of the patch (default [0 1 0])
%   'faceAlpha'  patch transparency ( defaul 0.2)
%   'edgeAlpha'  edge --//-- (default 0.2)
%   'fitAxis'    change axis to make the patch fully visible (default 1)
%
% Example:
%  load( 'riply_dataset', 'Trn' );
%  Trn.X  = [Trn.X ; Trn.X(1,:).* Trn.X(2,:)];
%  [w,w0] = svmocas( Trn.X,1,Trn.Y, 1);
%
%  figure;
%  ppatterns( Trn.X, Trn.Y);
%  hold on; grid on;
%  phyperplane(w,w0);
%


    if nargin >= 3, Opt = c2s( varargin ); else Opt = []; end
    if ~isfield( Opt, 'color'), Opt.color = [0 1 0]; end
    if ~isfield( Opt, 'viewAz'), Opt.viewAz = -45; end
    if ~isfield( Opt, 'viewEl'), Opt.viewEl = 45; end
    if ~isfield( Opt, 'faceAlpha'), Opt.faceAlpha = 0.2; end
    if ~isfield( Opt, 'edgeAlpha'), Opt.edgeAlpha = 0.2; end
    if ~isfield( Opt, 'fitAxis'), Opt.fitAxis = 1; end
    

    xp = get(gca,'Xlim');
    yp = get(gca,'Ylim');
    zp = get(gca,'Zlim');

    if w(3) ~= 0
        x = [ xp(1) xp(2) xp(2) xp(1)];
        y = [ yp(1) yp(1) yp(2) yp(2)];
        z = -(w0+w(1)*x+w(2)*y)/w(3);
    elseif w(2) ~= 0
        z = [ zp(1) zp(2) zp(2) zp(1)];
        x = [ xp(1) xp(1) xp(2) xp(2)];
        y = -(w0+w(1)*x+w(3)*z)/w(2);    
    elseif w(1) ~= 0
        z = [ zp(1) zp(2) zp(2) zp(1)];
        y = [ yp(1) yp(1) yp(2) yp(2)];
        x = -(w0+w(2)*y+w(3)*z)/w(1);    
    else
       error('all(w==0) is not allowd.');
    end

    h = patch(x,y,z, Opt.color);
    set(h,'facealpha',Opt.faceAlpha);
    set(h,'edgealpha',Opt.edgeAlpha);

    if Opt.fitAxis
        box1 = axis;
        box2 = [min(x) max(x) min(y) max(y) min(z) max(z)];
        box1 = reshape( box1, 2, length( box1 )/2);
        box2 = reshape( box2, 2, length( box2 )/2);
    
        newBox = [min( box1(1,:), box2(1,:)) ; max( box1(2,:), box2(2,:))];
        newBox = newBox(:);
        axis(newBox);
    end
    
    view(Opt.viewAz,Opt.viewEl);
end
% EOF