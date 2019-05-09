function str = mlp_param_str(Params)

   if isfield( Params, 'dropOut') && ~isempty(Params.dropOut) && Params.dropOut ~= 0
      str = sprintf('nr%d_bs%d_lr%f_do%f', Params.nLayers, Params.batchSize,...
          Params.learningRate, Params.dropOut);
   else
      str = sprintf('nr%d_bs%d_lr%f', Params.nLayers, Params.batchSize, Params.learningRate);
   end
end
