function Net = add_max_pool_do( Net, layer, inVar, outVar, Param, dropOutRate )
% MAX_POOL + DROP_OUT
% MAX_POOL
%
    
    input = inVar;
    
    if ~isempty( dropOutRate )
        output = [outVar '_xm'];

        Net.addLayer( [layer '_maxpool'], dagnn.Pooling('poolSize', ...
            Param.poolSize,'pad',Param.pad,'stride',Param.stride,'method',Param.method), ...
            input, output);

        input  = output;
        output = outVar;
        
        Net.addLayer( [layer '_dropout'], dagnn.DropOut('rate', dropOutRate ), input, output);
    else
        
        output = outVar;
        Net.addLayer( [layer '_maxpool'], dagnn.Pooling('poolSize', ...
            Param.poolSize,'pad',Param.pad,'stride',Param.stride,'method',Param.method), ...
            input, output );
    end
    

end

