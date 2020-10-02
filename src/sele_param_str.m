function str = sele_param_str(Params)

   str = sprintf('lambda%f_bs%d', Params.lambda, Params.batchSize );
end
