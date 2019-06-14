function z = preimage(SV, Alpha, kernelName, kernelArgs)
% PREIMAGE solves preimage problem for given kernel.
%
% Synopsis:
%   z = preimage(SV, Alpha, kernelName, kernelArgs)
%

switch lower(kernelName)
    %% POLYNOMIALS
    case 'poly'        
      if kernelArgs(1) == 2 && (length(kernelArgs) == 1 || kernelArgs(2) == 0)
        z = preimage_poly2(SV,Alpha);
      else
        error('Preimage implemented only for homogeneous degree 2 polynomials.');
      end
      
    %% RBF
    case 'rbf'
        z = preimage_rbf(SV,Alpha,kernelArgs);

    otherwise
        error('Preimage problem not implemented for given kernel.'); 
end

return;