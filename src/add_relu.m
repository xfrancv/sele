function Net = add_relu( Net, layer, inVar, outVar, Param)
%% Net = add_relu( Net, layer, inVar, outVar, Param)
%
    Net.addLayer( layer, dagnn.ReLU('leak', Param.leak), inVar, outVar);
end
