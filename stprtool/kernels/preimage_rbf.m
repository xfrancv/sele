function z = preimg_rbf(SV,Alpha,kernelArgs,Options)
% PREIMAGE_RBF RBF pre-image problem solved by gradient descent.
%
% Synopsis:
%   z = preimg_rbf(SV,Alpha,kernelArgs)
%   z = preimg_rbf(SV,Alpha,kernelArgs,Options)
%
%  Description:
%   This function solves the preimage problem for the RBF kernel
%   by means of gradient descent optimization. It exploits Matlab's
%   fminunc. 
% 
% Input:
%   SV [nDim x nSV] Support vectors.
%   Alpha [a x nSV] Weight vector. 
%   kernelArgs [1x1] Width of RFB kernel. See HELP KERNEL.
% 
%   Options [struct] 
%    .minImprov [1x1] If the improvement in two consecutive steps drops
%      below MinImprov the algorithm stops (default 1e-3). 
%    .numInitPoints [1x1] Number of random attempts to select inital 
%      point from SV (default 50).
%
% Output:
%  z [nDim x 1] Found preimage.
%
% See also 
%  TBA

%% process input arguments
if nargin < 4, Options = []; end
if ~isfield(Options,'minImprov'), Options.minImprov = 1e-3; end
if ~isfield(Options,'numInitPoints'), Options.numInitPoints = 50; end

INIT_STEP = 1e-3; 

[nDim,nSV] = size(SV);
iXi = sum( SV.^2)';

%% Select initial solution
rand_inx = randperm( nSV );
rand_inx = rand_inx(1:min([nSV,Options.numInitPoints]));
Z = SV(:,rand_inx);
fval = kernel(Z,SV,'rbf',kernelArgs)*Alpha(:);
fval = -fval.^2;
[dummy,idx] = min( fval );
z = Z(:,idx);

%% Start gradient descent optimization  
improvment=inf;
opt=optimset('display','off','Diagnostics','off','LargeScale','off');
warning off MATLAB:divideByZero;
while improvment > Options.minImprov
   
   % compute gradient
   dotp = kernel( SV, z, 'rbf', kernelArgs ) .* Alpha(:);
   dz = z*sum(dotp) - SV*dotp;

   % auxiciliary variables
   zXi = SV'*z;
   dzXi = SV'*dz;   
   Ai = -kernelArgs * (iXi - 2*zXi + z'*z);
   Bi = -kernelArgs * (z'*dz - dzXi);
   C  = -kernelArgs * (dz'*dz);
   
   % line-search
   [t,fval] = fminunc(@foo,INIT_STEP,opt,Alpha,Ai,Bi,C);
      
   % update solution
   old_z = z;
   z = z + dz*t;
      
   improvment = sum((z-old_z).^2);
end

return;

function f=foo(t,Alpha,Ai,Bi,C)

f = Alpha(:)'*exp(Ai + Bi*t + C*t^2);
f = -f^2;

return;
% EOF


