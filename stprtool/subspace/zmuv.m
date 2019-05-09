function Model = zmuv( X )
% ZMUV Affine map which makes points to have zero mean and unit variance
%
% Synopsis:
%   Model = zmuv( X )
%
% Examples:
%   X     = mvnrnd([-1;1],[0.2 0;0 2],100)';
%   Z     = affinemap( X, zmuv( X ) );
%   mean(Z,2)
%   std(Z,0,2)
% 

    m = mean( X, 2);
    s = std( X, 0, 2);
    s( find(s == 0) ) = 1;

    Model.W  = diag( 1./s );
    Model.W0 = -Model.W*m;

end