function Net = add_conv_bn_relu_do( Net, layer, inVar, outVar, Param, useBatchNorm, dropOutRate)
%%
% CONV + BATCH_NORM + RELU + DROP_OUT
% CONV              + RELU + DROP_OUT
% CONV              + RELU 
% CONV + BATCH_NROM + RELU 
%


    input  = inVar;
    output = [outVar '_conv'];

    % conv
    if( Param.bias == true)
        Net.addLayer( [layer '_cov'], ...
            dagnn.Conv('size',Param.f,'pad',Param.pad,'stride',Param.stride,'hasBias',Param.bias), ...
            input, output, { ['w_' layer], ['b_' layer]} );   
    else
        Net.addLayer( [layer '_cov'], ...
            dagnn.Conv('size',Param.f,'pad',Param.pad,'stride',Param.stride,'hasBias',Param.bias), ...
            input, output, {['w_' layer] });  
    end

    % batch norm
    if useBatchNorm
        depth  = Param.f(4);
        input  = output;
        output = [outVar '_bn'];
        
        Net.addLayer( [layer '_bn'], dagnn.BatchNorm('numChannels', ...
            depth, 'epsilon', 1e-5), input,  output,...
            { ['gg_'  layer], [ 'bb_' layer], ['mm_' layer] });
    end
    
    input  = output;

    if ~isempty( dropOutRate )
        % dropout + relu 
        
        output = [outVar '_xr'];
        Net.addLayer( [layer '_relu'], dagnn.ReLU('leak', Param.leak), input, output);

        input  = output;
        output = outVar;
        
        Net.addLayer( [layer '_dropout'], dagnn.DropOut('rate', dropOutRate ), input, output);
    else
        % relu
        output = outVar; 
        Net.addLayer( [layer '_relu'], dagnn.ReLU('leak', Param.leak), input, output);
    end
    
end
