function [commonX,commonY] = common_curve( X, Y)
% COMMON_CURVE put curves to a common domain X.
%
% Synopsis:
%  [commonX,commonY] = common_curve(X,Y)
%  
% Description:
%   X and Y are cell array. The pair X{i} and Y{i} are vectors
%   of the same length which describe a curve, i.e. set of points
%   (x,y).
%
% Examples:
%  x1 = [-2:0.1:10]; y1 = sin( x1 );
%  x2 = [-2:0.5:10]; y2 = cos( x2 );
%  [x,y] = common_curve({x1,x2},{y1,y2});
%
%  figure;
%  plot(x1,y1,'r'); hold on;
%  plot(x2,y2,'k');
%  errorbar(x,mean(y),std(y), -std(y));
%

nCurves = numel( X );

commonX = [];
uniqY   = [];
uniqX   = [];
nCurves = numel(X);

for i=1:nCurves    
    
    uniqX{i} = unique( X{i} );
    uniqY{i} = zeros(size( uniqX{i}));
    for j=1:length(uniqX{i})
        uniqY{i}(j) = mean( Y{i}( find( X{i}==uniqX{i}(j))));
    end
    
    commonX = [commonX X{i}(:)'];
end

commonX = unique( commonX );
commonY = zeros( nCurves, length( commonX ) );

for i = 1 : nCurves
   commonY(i,:) = interp1( uniqX{i}, uniqY{i}, commonX, 'linear' )';
end

return;
