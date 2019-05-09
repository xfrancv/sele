function out_obj = kernel_get_data(in_obj,idx,kernel_name)
% KERNEL_GET_DATA
%
% Synopsis:
%   out_obj = kernel_get_data(in_obj,idx,kernel_name)
%

switch kernel_name
    case {'linear','poly','rbf','sigmoid'} 
        out_obj = in_obj(:,idx);
                        
    otherwise
        error('Unknown kernel.');
end

return;
%EOF