function h=pline(W,W0,line_style)
% PLINE Plots line in 2D.
%
% Synopsis:
%  h=pline(W,W0)
%  h=pline(W,W0,line_style)
%
% Description:
%  The line to be plotted can be described either
%
%  implicitely by W'*x + W0 = 0 where W [2x1], W0[1x1] and x [2x1], or
%
%  explicitely by y = W*x + W0 where W, W0, x and y are scalers.
%
% Input:
%  W, W0 see above.
%  line_style [struct] See HELP PLOT. By default 'k-'.
%
% Output:
%  h [1x1] Handle to the line.
%
% Example:
%  figure; hold on; axis([-1 1 -1 1]);
%  pline(inf,0,'--'); 
%  pline(0,0,'--');
%  pline([1;1],0);
%

% About: Statistical Pattern Recognition Toolbox
% (C) 1999-2003, Written by Vojtech Franc and Vaclav Hlavac
% <a href="http://www.cvut.cz">Czech Technical University Prague</a>
% <a href="http://www.feld.cvut.cz">Faculty of Electrical Engineering</a>
% <a href="http://cmp.felk.cvut.cz">Center for Machine Perception</a>

% Modifications:
% 29-apr-2004, VF
% 13-july-2003, VF
% 20-jan-2003, VF
% 8-jan-2003, VF, A new coat.

hold_status=ishold;
hold on;

if nargin < 3,
    line_style = 'k-';
end

if length(W)==1,
   W = [W; -1];
end

%%
win = axis;
[x1,y1,x2,y2,in]=clipline(W,W0,win);
h = plot([x1,x2],[y1,y2],line_style);

%%
if nargout >0
    varargout{1}= h; 
end

if ~hold_status
    hold off; 
end

return;

