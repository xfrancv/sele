function str = svm_param_str(Params)

    if ~isfield(Params,'rbfWidth')
        if isfield( Params,'C')
            str = sprintf('C%f', Params.C );
        else
            str = sprintf('lambda%f', Params.lambda);
        end
    else
        str = sprintf('width%f_C%f', Params.rbfWidth, Params.C );
    end
end
