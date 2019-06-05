function setup_matconvnet(rootFolder)
%% setup SETPATH
%

   %% get root folder
   if nargin < 1
       rootFolder = which('run_train_lr.m');
       rootFolder = [fileparts( rootFolder ) '/'];
   end
   
   
   run([rootFolder 'stprtool/stprpath.m']);
   
   addpath( rootFolder );
   addpath( [rootFolder 'src/']);
   addpath( [rootFolder 'src/libsvm-3.1/']);
   addpath( [rootFolder 'libocas/']);
   
   
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
