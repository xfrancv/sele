function Net = add_pool( Net, layer, inVar, outVar, Param)
% Net = add_pool( Net, layer, inVar, outVar, Param)
    
    input = inVar;
    
    output = [outVar '_xm'];

    Net.addLayer( layer , dagnn.Pooling('poolSize', Param.poolSize,...
        'pad',Param.pad,'stride',Param.stride,'method',Param.method), ...
            inVar, outVar);    

end

