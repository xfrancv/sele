function h = pclassifier(Model,Classifier,Opts)
% PCLASSIFIER Plots decision boundary of a classifier.
%
% Synopsis:
%  h = pclassifier(model)
%  h = pclassifier(model,classifier)
%  h = pclassifier(model,classifier,options)
%  h = pclassifier(model,[],options)
%
% Description:
%  This function visualizes a decision boundary of a classifier whose
%  parameters are encapsulated in the argument "model". The classifier to 
%  be visualized is determined by a function handler passed in the argument
%  "classifier". If the argument "classifier" is empty, the "model.eval"
%  is used instead. 
%  
%  The classifier function must have the following calling syntax
%    y = classifier( X, model )
%  where y [nExamples x 1] are labels [1:maxLabel] and X [nDim x nExamples]
%  are input examples (pclassifier uses nDim=2). 
%
% Input:
%  model [struct] Encapsulates parameters of the classifier. 
%  classifier [function_handle] Handle of the classifier function (see
%     Description above). If classifier is [] then pclassifier uses
%     model.eval instead. 
%  options [struct] Controls the way of visialization:
%   .gridx [1x1] Sampling density in x-axis (default 300).
%   .gridy [1x1] Sampling density in y-axis (default 300).
%   .LineSpec [string] specifies the line type and color of the contour.
%   .fill [1x1] If 1 then the class regions are filled. 
%  
% Output:
%  h [1 x nobjects] Handles of graphics objects used. 
%
% Example:
%  KNN   = load('fiveclassproblem','X','Y');
%  KNN.K = 1;
%  figure;
%  ppatterns( KNN.X, KNN.Y );
%  pclassifier( KNN, @knnclassif );
%
% See also 
%  PPATTERNS, PLINE.
%

COLORS  = ['brgckmywbrgc'];

%% 
if nargin < 1
    error('Not enough input arguments.'); 
end

if nargin < 3, Opts=[]; end
if ~isfield(Opts,'fill'), Opts.fill=0; end
if ~isfield(Opts,'gridx'), Opts.gridx = 300; end
if ~isfield(Opts,'gridy'), Opts.gridy = 300; end
if ~isfield(Opts,'LineSpec'), Opts.LineSpec = 'k-'; end

colorMap = [ 
    [255 204 204]/255 ; 
    [178 255 102]/255; 
    [102 178 255]/255; 
    [204 204 255]/255;
    [255 153 153]/255; 
    [229 255 204]/255 ;
    [255 255 153]/255];

%% 
holdStatus = ishold;
hold on;

%% create grid 
V     = axis;
x     = linspace(V(1),V(2),Opts.gridx);
y     = linspace(V(3),V(4),Opts.gridy);
[X,Y] = meshgrid(x,y);

%% create test points
tstX = [X(:)'; Y(:)'];
%tst_X=[reshape(X',1,prod(size(X)));reshape(Y',1,prod(size(Y)))];

%% classify test points
if nargin < 2 || isempty(Classifier)
    D = Model.eval( tstX, Model);
else
    D = Classifier( tstX, Model);
end

%%
Z  = reshape( D, Opts.gridx,Opts.gridy);
h1 = image(x,y,Z);
h1.AlphaData = 0.5;
if max(D(:)) < size(colorMap,1)
    colormap( colorMap );
end

%%
[~,h2] = contour(x,y,Z);

h2.LineColor='k';
h2.LineWidth = 2;
%%
if ~holdStatus
    hold off; 
end

%% return handles if required
if nargout > 0
    varargout{1} = [h1 h2]; 
end

return;

%-------------------------------------------------------
function h = plot_boundary( L, X_pos, Y_pos, fill_regions, linestyle )
% Plots decision boudary. 
%
COLORS  = ['brgckmywbrgc'];

dx = X_pos(2)-X_pos(1);
dy = Y_pos(2)-Y_pos(1);
m = length( X_pos );
n = length( Y_pos );

Z = NaN*ones( m+2, n+2 );
num_classes = max(L(:));

% mask=fspecial('gauss',[5 5],1) ; % fspecial is from images toolbox 
mask = [0.0030    0.0133    0.0219    0.0133    0.0030;
        0.0133    0.0596    0.0983    0.0596    0.0133;
        0.0219    0.0983    0.1621    0.0983    0.0219;
        0.0133    0.0596    0.0983    0.0596    0.0133;
        0.0030    0.0133    0.0219    0.0133    0.0030];

h = [];      
for i = 1:num_classes,
  
  A=L;
  A(find(L==i))=1;
  A(find(L~=i))=-1;
    
  A = reshape( A', m, n );
  A = filter2(mask, A);
   
  Z(2:end-1,2:end-1) = A;  
  
   [cc,tmp_h] = contourf([X_pos(1)-dx,X_pos,X_pos(end)+dx],...
                    [Y_pos(1)-dy,Y_pos,Y_pos(end)+dy],Z');  
%   [cc,tmp_h] = contour([X_pos(1)-dx,X_pos,X_pos(end)+dx],...
%                    [Y_pos(1)-dy,Y_pos,Y_pos(end)+dy],Z',[-0 0],linestyle);  
  h = [h tmp_h(:)'];
  
  if fill_regions,
   while ~isempty(cc)
     len = cc(2,1);
     tmp_h = fill(cc(1,2:len+1),cc(2,2:len+1),COLORS(i));
     h = [h tmp_h(:)'];
     cc(:,1:len+1) = [];
   end  
  end
end

return;
% EOF