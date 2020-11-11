function Net = init_confnet1(nInputs, nOutputs, nHiddenStates, varargin )
% 
% Net = init_confnet1(nInputs, nOutputs, nHiddenStates, varargin )
%


    %%
    opts.useBatchNorm = true;
    opts.dropOutRate  = [];
    opts.lab2age      = [];
    opts.lab2gender   = [];
    opts.leak         = 0;
    opts.loss         = 1;
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
    switch opts.loss
        case 1 % sele1 - hinge-loss
            Net.addLayer('rankloss', dagnn.LossRank(), {'prediction','risk','predY'}, 'objective' );
        case 2 % sele2 - logistic function
            Net.addLayer('rankloss', dagnn.LossRankLog(), {'prediction','risk','predY'}, 'objective' );
        case 3 % sele3 - hing-loss - no quadratic comparison
            Net.addLayer('rankloss', dagnn.LossSele(), {'prediction','risk','predY'}, 'objective' );
    end
    
    Net.addLayer('auRC', dagnn.LossAuRC(), {'prediction','risk','predY'}, 'auRC' );
    
    %%
    Net.initParams();
end