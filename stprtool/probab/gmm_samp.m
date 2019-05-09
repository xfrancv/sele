function [X,Y] = gmm_samp(model,nSamples)
% GMM_SAMP Samples random vectors from Gaussian Mixture Model.
% 
% Synopsis:
%  [X,Y] = gmm_samp(model,nSamples)
%
% Description:
%  This function generates nSamples random vector from the given GMM.
%
% Input:
%  model [struct] GMM (see HELP GMM_ML).
%  N [1x1]        Number of vectors to be generated.
%
% Output:
%  X [nDim x nSamples] Random samples.
%  Y [1 x nSamples]    Labels of the Gassian components.
%
% Example:
%  % 1D example
%  Mean = [-2 2]; Var = [1 0.5]; Prior = [0.4 0.6];
%  model = gmm_create(Mean,Var,Prior);
%  figure; hold on; 
%  plot([-5:0.1:5], exp(gmm_logpx([-5:0.1:5],model)),'r');
%  [Y,X] = hist(gmm_samp(model,1000),25);
%  bar(X,Y/(1000*(X(2)-X(1))));
%
%  % 2D example
%  Mean = [-1 1;-1 1]; Prior = [0.4 0.6];
%  Cov(:,:,1)=[.1 0; 0 1]; Cov(:,:,2)=[1 0;0 .1];
%  model = gmm_create(Mean,Cov,Prior);
%  figure; hold on; 
%  [X,Y] = gmm_samp(model,200);
%  ppatterns(X,Y);
%  pgmm(model);
%
% See also
 
% get dimensions
[nDim,nComp] = size(model.Mean);

% generate labels of the Gaussian componenents 
Y = d_samp(model.Prior,nSamples);

% generate vectors from the Gaussian components Y
X = zeros(nDim,nSamples);
for y=1:nComp
    
  idx = find(Y==y);
  N = length(idx);
  if N > 0
    
      switch model.covType
          case 'full'
              R = model.U(:,:,y)'*diag(sqrt(model.D(:,y)));
          case 'diag'
              R = diag(sqrt(model.D(:,y)));
              
          case 'spherical'
              R = eye(nDim,nDim)*sqrt(model.D(y));
      end
      X(:,idx) = R*randn(nDim,N) + repmat(model.Mean(:,y),1,N);
  end
end

return;
