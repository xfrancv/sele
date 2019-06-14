function X = pcarec(Y, PcaModel)
% PCAREC Reconstuct points projected by PCA model.
%
% X = pcarec(Y, PcaModel)
%
% Example:
%    X=mvnrnd([1;1],[0.13 0.2;0.2 0.4],1000)';  % generate 2d points
%    PcaModel = pca(X,1);                       % find PCA model
%    Y = affinemap(X, PcaModel);                % extract principal components
%    Xrec = pcarec(Y,PcaModel);
%    figure; 
%    ppatterns(X); hold on;
%    plot(Xrec(1,:),Xrec(2,:),'r+');
%    axis equal;
% 

X = PcaModel.W*Y+repmat(PcaModel.mean,1,size(Y,2) );

return;
% EOF