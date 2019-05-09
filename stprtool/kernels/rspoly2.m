function ReducedModel = rspoly2( Model, newNumSV )
% RSPOLY2 Reduced set method for homogeneous degree two polynomial.
%
% Synopsis:
%  ReducedModel = rspoly2( Model )
%  ReducedModel = rspoly2( Model, newNumSV )
%
% Description:
%  This function implements reduced set method for reducing the number of
%  support vectors in kernel expansion over homogeneous degree two
%  polynomial kernels 
%     k(x,y) = (x'*y)^2 = kernel(x,y,'poly',[2 0 1]) 
%
%  Given kernel expansion
%   Phi1 =   sum   Model.Alpha(i)*Phi(Model.SV(:,i)) 
%          i=1:nSV
%  the goal is to find ReducedModel such that the distance ||Phi1-Phi2|| 
%  is minimal where 
%   Phi2 =   sum   ReducedModel.Alpha(i)*Phi(ReducedModel.SV(:,i))
%        i=1:new_nSV     
%  is the reduced kernel expansion and Phi is given implictily by
%  Phi(x)'*Phi(y) = (x'*y)^2. 
%  
%  The global optimum is found by method published in 
%  J.C.Burges: Simplified Support Vector Decision Rules. ICML, 1996.
%  
% Input:
%  Model [struct] Kernel expansion:
%   .Alpha [1 x nSV] Kernel weights.
%   .bias [1x1] Bias.
%   .SV [nDim x nSV] Support vectors.
%   .kernelAame = 'poly'
%   .kernelArgs = [2 0 1]
%
%  newNumSV [1x1] Maximal number of support vectors in the reduced kernel 
%    expansion. If new_nSV is not given then the new expansion approximates 
%    the original one exactly with at most nDim support vectors. 
%
% Output:
%  ReducedModel [struct] Reduced kernel expansion:
%   .Alpha [1 x newNumSV] New kernel weights.
%   .bias [1x1] Bias (unchanged).
%   .SV [nDim x newNumSV] New support vectors.
%   .kernelName = 'poly'
%   .kernelArgs = [2 0 1]
%
% Example:
%  TBA
% See also 
%  TBA
%

% check inputs
if nargin < 2, newNumSV = inf; end

if strcmpi(Model.kernelName,'poly') ~= 1 | ...
   (Model.kernelArgs ~= [2 0 1] & ...
    Model.kernelArgs ~= [2 0] & ...
    Model.kernelArgs ~= 2)
  error('Kernel must be homogeneous degreee two polynomial.');
end

[nDim,nSV] = size( Model.SV);
S = zeros(nDim,nDim);

for i=1:nDim,
  for j=i:nDim,
    S(i,j) = (Model.SV(i,:).*Model.SV(j,:) )*Model.Alpha(:);
    S(j,i) = S(i,j);
  end
end

[V,D] = eig(S);
D = real(diag(D));
[dummy,inx] = sort(-abs(D));
D=D(inx);
V=V(:,inx);
inx = find(D ~= 0);

% take at most new_nSV support vectors
newNumSV = min(length(inx),newNumSV);
inx = inx(1:newNumSV);

ReducedModel.Alpha = zeros(1,newNumSV);
ReducedModel.bias = Model.bias;
ReducedModel.kernelName = 'poly';
ReducedModel.kernelArgs = [2 0 1];
ReducedModel.SV = zeros(nDim,newNumSV);
ReducedModel.eval = @kernelclassif;

cnt = 0;
for i=inx(:)',
  cnt = cnt+1;
  ReducedModel.SV(:,cnt) = V(:,i);
  ReducedModel.Alpha(cnt) = D(i)/...
      (ReducedModel.SV(:,cnt)'*ReducedModel.SV(:,cnt));
end

return;
% EOF