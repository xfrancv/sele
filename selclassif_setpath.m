function setup_matconvnet(rootFolder)
%% setup SETPATH
%

   %% get root folder
   if nargin < 1
       rootFolder = which('selclassif_install.m');
       rootFolder = [fileparts( rootFolder ) '/'];
   end
   
   
   run([rootFolder '3rdparty/stprtool/stprpath.m']);
   
   addpath( rootFolder );
   addpath( [rootFolder 'src/']);
   addpath( [rootFolder 'ordreg/src/']);
   addpath( [rootFolder '3rdparty/libsvm-3.1/']);
   addpath( [rootFolder '3rdparty/libocas/']);
   
   
    switch hostname
        case 'halmos'
            matconvnet_sufix = '_halmos';
        case 'boruvka'
            matconvnet_sufix = '_boruvka';
        case 'lcgpu2'
            matconvnet_sufix = '_lcgpu2';
        case 'goedel'
            matconvnet_sufix = '_goedel';        
        otherwise % for computer without GPU
            matconvnet_sufix = '';        
    end
   
   
   run(  [rootFolder 'matconvnet-1.0-beta25' matconvnet_sufix '/matlab/vl_setupnn.m']);

end
