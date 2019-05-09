function Net = add_conv_bn( Net, layer, inVar, outVar, Param, useBatchNorm)
% CONV + BATCH_NORM
% CONV
%

    input = inVar;
    
    if useBatchNorm
        output = sprintf('xc%d', layer);

        if( Param.bias == true)
            Net.addLayer( [layer '_conv'], ...
                dagnn.Conv('size',Param.f,'pad',Param.pad,'stride',Param.stride,'hasBias',Param.bias), ...
                input, output, { ['w_' layer], ['b_' layer]});   
        else
            Net.addLayer( [layer '_conv'], ...
                dagnn.Conv('size',Param.f,'pad',Param.pad,'stride',Param.stride,'hasBias',Param.bias), ...
                input, output, { ['w_' layer ]});  
        end
        
        depth  = Param.f(4);
        input  = output;
        output = outVar;
        
        Net.addLayer( [layer '_bn'], dagnn.BatchNorm('numChannels', ...
            depth, 'epsilon', 1e-5), input,  output,...
            { ['gg_' layer], [ 'bb_' layer], ['mm_' layer]});

    else
        output = outVar;
        
    
        if( Param.bias == true)
            Net.addLayer( [layer '_conv'], ...
                dagnn.Conv('size',Param.f,'pad',Param.pad,'stride',Param.stride,'hasBias',Param.bias), ...
                input, output, { ['w_' layer], ['b_' layer]});   
        else
            Net.addLayer( [layer '_conv'], ...
                dagnn.Conv('size',Param.f,'pad',Param.pad,'stride',Param.stride,'hasBias',Param.bias), ...
                input, output, { ['w_' layer ]});  
        end
    end    
end
