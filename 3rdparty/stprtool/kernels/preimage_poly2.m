function z = preimage_poly2(SV,Alpha)
% PREIMAGE_POLY2 solves preimage problem for homogeneous degree 2 polynomial.
%
% Synopsis:
%   z = preimage_poly2(SV,Alpha)
%
% Input:
%  SV [nDims x nSV] Suport vectors.
%  Alpha [1 x nSV] Weights.
%
% Output:
%  z [nDim x 1] Found preimage.
%

nDim = size(SV,1);
S = zeros(nDim,nDim);

for i=1:nDim,
  for j=i:nDim,
    S(i,j) = (SV(i,:).*SV(j,:) )*Alpha(:);
    S(j,i) = S(i,j);
  end
end

[V,D] = eig(S);
[dummy,idx] = max(abs(diag(D)));
z = V(:,idx);

return;
