function stprcompile(root)
% STPRCOMPILE Compiles all MEX functions contained in STPRtoolbox.
%
% Synopsis:
%  stprcompile
%  stprcompile( toolboxroot )
%
% Description:
%  It calls MEX complier on all C an Fortran codes. 
%  Run this function from the STPRtoolbox root folder or 
%  you specify the root directory as an input argument.
%

fprintf('Compiling MEX files of STPRtool...\n');

if nargin < 1
   root=pwd;              % get current directory
end

%% List of functions to be complied 

% KNNEST
fun(1).source={'$misc/knnest_mex.c'};
fun(1).outdir='$misc';
fun(1).output='knnest';
fun(1).include=[];
fun(1).flags='-largeArrayDims';

fun(2).source={'$kernels/kernel_mex.c','$kernels/kernel_functions.c'};
fun(2).outdir='$kernels';
fun(2).output='kernel';
fun(2).include=[];
fun(2).flags=[];

fun(3).source={'$optimization/libqp/libqp_splx_mex.c','$optimization/libqp/libqp_splx.c'};
fun(3).outdir='$optimization/libqp';
fun(3).output='libqp_splx_mex';
fun(3).include=[];
fun(3).flags='-largeArrayDims';

fun(4).source={'$optimization/libqp/libqp_gsmo_mex.c','$optimization/libqp/libqp_gsmo.c'};
fun(4).outdir='$optimization/libqp';
fun(4).output='libqp_gsmo_mex';
fun(4).include=[];
fun(4).flags='-largeArrayDims';

fun(5).source={'$svm/msvmb2_mex.c','$optimization/libqp/libqp_splx.c','$kernels/kernel_functions.c'};
fun(5).outdir='$svm';
fun(5).output='msvmb2_mex';
fun(5).include={'$optimization/libqp','$kernels'};
fun(5).flags=[];

fun(6).source={'$kernels/kernelmap_mex.c','$kernels/kernel_functions.c'};
fun(6).outdir='$kernels';
fun(6).output='kernelmap_mex';
fun(6).include=[];
fun(6).flags=[];


fun(7).source={'$misc/mult_matrices_int32xint32_mex.c'};
fun(7).outdir='$misc';
fun(7).output='mult_matrices_int32xint32_mex';
fun(7).include=[];
fun(7).flags='-largeArrayDims';

fun(8).source={'$misc/mult_matrices_doublexint8_mex.c'};
fun(8).outdir='$misc';
fun(8).output='mult_matrices_doublexint8_mex';
fun(8).include=[];
fun(8).flags='-largeArrayDims';

fun(9).source={'$misc/mult_matrices_doublexsparselogical_mex.c'};
fun(9).outdir='$misc';
fun(9).output='mult_matrices_doublexsparselogical_mex';
fun(9).include=[];
fun(9).flags='-largeArrayDims';

fun(10).source={'$misc/mult_matrices_int32xdouble_mex.c'};
fun(10).outdir='$misc';
fun(10).output='mult_matrices_int32xdouble_mex';
fun(10).include=[];
fun(10).flags='-largeArrayDims';

fun(11).source={'$features/liblbp_v2.cpp','$features/lbppyr_mex.cpp'};
fun(11).outdir='$features';
fun(11).output='lbppyr';
fun(11).include=[];
fun(11).flags='-largeArrayDims';

fun(12).source={'$features/liblbp_v2.cpp','$features/ltppyr_mex.cpp'};
fun(12).outdir='$features';
fun(12).output='ltppyr';
fun(12).include=[];
fun(12).flags='-largeArrayDims';

fun(13).source={'$features/liblbp_v2.cpp','$features/ltppyrsparse_mex.cpp'};
fun(13).outdir='$features';
fun(13).output='ltppyrsparse';
fun(13).include=[];
fun(13).flags='-largeArrayDims';

fun(14).source={'$features/liblbp_v2.cpp','$features/lbppyrsparse_mex.cpp'};
fun(14).outdir='$features';
fun(14).output='lbppyrsparse';
fun(14).include=[];
fun(14).flags='-largeArrayDims';

fun(end+1).source={'$features/liblbp_v2.cpp','$features/unilbppyr_mex.cpp'};
fun(end).outdir='$features';
fun(end).output='unilbppyr';
fun(end).include=[];
fun(end).flags='-largeArrayDims';

fun(end+1).source={'$optimization/lbfgs/lbfgs_mex.cpp','$optimization/lbfgs/lbfgs.cpp'};
fun(end).outdir='$optimization/lbfgs';
fun(end).output='lbfgs';
fun(end).include=[];
fun(end).flags='-largeArrayDims';

fun(end+1).source={'$optimization/maxsum/maxsum_feasible_mex.c'};
fun(end).outdir='$optimization/maxsum';
fun(end).output='maxsum_feasible';
fun(end).include=[];
fun(end).flags='';

%% Run MEX compiler
for i=1:length(fun),    
       
    mexstr = ['mex '...
              fun(i).flags ...
              ' ' ...
              ' -O' ...
              ' -outdir ' translate(fun(i).outdir, root) ...
              ' -output ' fun(i).output];
    for j=1:length(fun(i).include)
        mexstr = [mexstr ' -I' translate(fun(i).include{j},root)];
    end
    
    mexstr = [mexstr ' '];
  
    for j=1:length(fun(i).source),    
        mexstr = [mexstr translate(char(fun(i).source(j)),root) ' '];
    end

    fprintf('%s\n',mexstr);
  
    eval(mexstr);
end

fprintf('Done.\n');

return;

%--translate ---------------------------------------------------------
function p = translate(p,toolboxroot);
%TRANSLATE Translate unix path to platform specific path
%   TRANSLATE fixes up the path so that it's valid on non-UNIX platforms
%
% This function was derived from MathWork M-file "pathdef.m"

cname = computer;
% Look for VMS, this covers VAX_VMSxx as well as AXP_VMSxx.
%if (length (cname) >= 7) & strcmp(cname(4:7),'_VMS')
%  p = strrep(p,'/','.');
%  p = strrep(p,':','],');
%  p = strrep(p,'$toolbox.','toolbox:[');
%  p = strrep(p,'$','matlab:[');
%  p = [p ']']; % Append a final ']'

% Look for PC
if strncmp(cname,'PC',2)
  p = strrep(p,'/','\');
  p = strrep(p,':',';');
  p = strrep(p,'$',[toolboxroot '\']);

% Look for MAC
elseif strncmp(cname,'MAC',3)
  p = strrep(p,':',':;');
  p = strrep(p,'/',':');
  m = toolboxroot;
  if m(end) ~= ':'
%    p = strrep(p,'$',[toolboxroot ':']); % dkim@mrn.org
    p = strrep(p,'$',[toolboxroot '/']);
  else
    p = strrep(p,'$',toolboxroot);
  end
else
  p = strrep(p,'$',[toolboxroot '/']);
end
