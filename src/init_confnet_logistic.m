function Net = init_confnet_logistic(nInputs, nOutputs, nHiddenStates, varargin )
% 
% Net = init_confnet1(nInputs, nOutputs, nHiddenStates, varargin )
%


    %%
    opts.useBatchNorm = true;
    opts.dropOutRate  = [];
    opts.leak         = 0;
    opts = vl_argparse(opts, varargin) ;

    rng('default');
    rng(0);

    %%
    Net = dagnn.DagNN();
    
    if numel( nHiddenStates ) > 0
    
        %% layer = 1;
        Param = struct('f', [1 1 nInputs nHiddenStates(1)], 'pad',  0, 'stride', 1, 'bias', true, 'leak', opts.leak);
        Net   = add_conv_bn_relu_do( Net, 'layer1', 'input', 'x1', Param, opts.useBatchNorm, opts.dropOutRate );


        %% layer 2 - numel(innerDims)
        for l = 2 : numel( nHiddenStates )
            Param = struct('f', [1 1 nHiddenStates(l-1) nHiddenStates(l)], 'pad',  0, 'stride', 1, 'bias', true, 'leak', opts.leak);
            Net   = add_conv_bn_relu_do( Net, ['layer' num2str(l)], ['x' num2str(l-1)], ['x' num2str(l)], Param, opts.useBatchNorm, opts.dropOutRate );
        end


        %% layer numel(innerDims)+2
        Param = struct('f', [1 1 nHiddenStates(end) nOutputs], 'pad',  0, 'stride', 1, 'bias', true, 'leak', opts.leak);
        Net   = add_conv_bn_relu_do( Net, 'output_layer', ['x' num2str(numel(nHiddenStates))], 'prediction', Param, opts.useBatchNorm, opts.dropOutRate );
        
    else
        %% single layer
        Param = struct('f', [1 1 nInputs nOutputs], 'pad',  0, 'stride', 1, 'bias', true, 'leak', opts.leak);
        Net   = add_conv_bn_relu_do( Net, 'output_layer', 'input', 'prediction', Param, opts.useBatchNorm, opts.dropOutRate );
    end
    
    
    %% Loss functions
    Net.addLayer('rankloss', dagnn.LossLogisticSwitch(), {'prediction','loss','predY'}, 'objective' );
    
    %%
    Net.initParams();
end