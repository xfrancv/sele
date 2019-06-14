function Y = pcaproj(X,Model)
% PCAPROJ Project input vector onto affice subspace defined by PCA model.
% Synopsis:
%  Y = pcaproj( X, PcaModel )
%
% Description:
%  PcaModel defines an affine linear subspace. This function projects
%  vectors in matrix X onto the subspace.
%
% Input:
%  X        [nDims x nExamples] Input vectors.
%  PcaModel [struct] see 'help pca'
%
% Output:
%  Y [nDims x nExamples] Reconstructed vectors.
%
% Example:
%   X        = mvnrnd([1;1],[0.13 0.2;0.2 0.4],100)';  % generate 2d points
%   PcaModel = pca(X,1);                       % find pca model
%   Y        = pcaproj(X, PcaModel );
%   figure; 
%   ppatterns(X); axis equal; grid on; hold on;
%   ppatterns(Y,[],'colors','r');
%   plot([X(1,:) ; Y(1,:);], [X(2,:); Y(2,:)],'k');
%   
%
% See also 
%  LINPROJ, PCA, KPCAREC.
%

% About: Statistical Pattern Recognition Toolbox
% (C) 1999-2003, Written by Vojtech Franc and Vaclav Hlavac
% <a href="http://www.cvut.cz">Czech Technical University Prague</a>
% <a href="http://www.feld.cvut.cz">Faculty of Electrical Engineering</a>
% <a href="http://cmp.felk.cvut.cz">Center for Machine Perception</a>

% Modifications:
% 25-may-2004, VF
% 5-may-2004, VF
% 22-apr-2004, VF
% 17-mar-2004, VF, created.

[~,nExamples] = size(X);

Y = Model.W*affinemap(X, Model) + Model.mean*ones(1,nExamples);

return;  
% EOF  
