%% compile stprtool
cd stprtool;
stprcompile;
cd ..;

%% compile libocas
%cd libocas;
%libocascompile;
%cd ..;

%% Download and compile MATCONVNET toolbox for training NN
switch hostname
    case 'halmos'
        matconvnet_sufix = '_halmos';
        use_gpu = true;
        opts.install.cuda_path = '/usr/local/cuda-9.1/';      % instalace na halmos
    case 'boruvka'
        matconvnet_sufix = '_boruvka';
        use_gpu = true;
        opts.install.cuda_path = '/usr/local/cuda-9.1/';      % instalace na halmos
    case 'lcgpu2'
        matconvnet_sufix = '_lcgpu2';
        use_gpu = true;
        opts.install.cuda_path = '/usr/local/cuda-9.0/';      % instalace na halmos
    case 'goedel'
        matconvnet_sufix = '_goedel';        
        use_gpu = true;
        opts.install.cuda_path = '/usr/local/cuda-9.1/';   % instalace na goedel
    otherwise % for computers without GPU
        matconvnet_sufix = '';        
        use_gpu = false;
        opts.install.cuda_path = [];        
end

matlabDir  = ['matconvnet-1.0-beta25' matconvnet_sufix];

% download if needed
if ~exist( matlabDir )
    untar('http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta25.tar.gz');
    system( ['mv matconvnet-1.0-beta25 ' matlabDir]);
end


% Copy customized layers
src = 'src/matlab/*';
dst = [ matlabDir '/matlab/'];
copyfile(src, dst,'f');

%
opts.install.matconvnet_path = [matlabDir '/matlab/vl_setupnn.m'];
run(opts.install.matconvnet_path) ;

if use_gpu
    vl_compilenn('enableGpu', true,'cudaRoot', opts.install.cuda_path, 'cudaMethod', 'nvcc') ;
else
    vl_compilenn('enableGpu', false);
end