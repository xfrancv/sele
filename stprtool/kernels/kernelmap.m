function proj = kernelmap( data, Model )
% KERNELMAP Kernel map.
%
% Synopsis:
%  proj = kernelmap( data, Model)
%
% Description:
%  This function maps input DATA onto real vectors using 
%
%  proj = Alpha*kernel(SV,data,kernelName,kernelArgs) + repmat(bias,1,N)
%
%  where parameters of the projection are given in the structure MODEL:
%     .Alpha [nOutDims x nSV] Multipliers.
%     .bias [nOutDims x 1] Bias.
%     .SV [array of nSV objects] Support vectors.
%     .kernelName [string] Kernel identifier.
%     .kernelArgs [...] Kernel arguments.
%
% Input:
%  data [array of N objects] Data to be projected.
%  model [struct] See above.
%  
% Output:
%  proj [nOutDims x N] Projections of input data. 
%
% Example:
%  TBA
% See also 
%  TBa
    
proj = kernelmap_mex( data, Model.Alpha, Model.bias, ...
        Model.SV, Model.kernelName, Model.kernelArgs );
    
return;
% EOF