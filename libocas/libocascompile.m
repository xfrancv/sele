function libocascompile(root)
% LIBOCASCOMPILE
%

fprintf('Compiling MEX files of LIBOCAS...\n');

if nargin < 1
   root=pwd;              % get current directory
end

%% List of functions to be complied 

% SVMOCAS
fun(1).source={'$svmocas_mex.c', '$ocas_helper.c', ...
    '$lib_svmlight_format.c', '$features_int8.c', ...
    '$features_double.c', '$features_single.c', ...
    '$libocas.c', '$libqp_splx.c'};
fun(1).outdir='./';
fun(1).output='svmocas';
fun(1).include=[];
fun(1).flags='-largeArrayDims';

% MSVMOCAS
fun(2).source={'$msvmocas_mex.c', '$lib_svmlight_format.c', ...
    '$ocas_helper.c', '$features_double.c', '$libocas.c', '$libqp_splx.c'};
fun(2).outdir='./';
fun(2).output='msvmocas';
fun(2).include=[];
fun(2).flags='-largeArrayDims';


%% Run MEX compiler
for i=1:length(fun),    
       
    mexstr = ['mex '...
              fun(i).flags ...
              ' ' ...
              ' -O' ...
              ' -DLIBOCAS_MATLAB' ...
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
