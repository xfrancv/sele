function model = gmm_create(Mean,Cov,Prior)
% GMM_CREATE Creates Gaussian Mixture Model from given parameters.
%
% Synopsis:
%    model = gmm_create(Mean,Cov,Prior)
%
%

[nDim,nComp] = size(Mean);

if nDim==1
    model.Mean = Mean;
    model.D = Cov(:);
    model.Prior = Prior;
    model.covType = 'spherical';
else
    model.Mean = Mean;
    model.U = zeros(nDim,nDim,nComp);
    model.D = zeros(nDim,nComp);
    for y=1:nComp
        [U S V] = svd(Cov(:,:,y));
        model.U(:,:,y) = U;
        model.D(:,y) = diag(S);
    end
    model.Prior = Prior;
    model.covType = 'full';
end

return;
% EOF