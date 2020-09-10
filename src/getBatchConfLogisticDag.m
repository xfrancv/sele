function inputs = getBatchLogisticDag(Opts,ImDb, batch, onlyInput)
%% 
     if nargin < 4
            onlyInput = 0;
     end
    
    images = single( ImDb.images.data(:,:,:,batch) );

    if ~isempty(Opts.gpus) > 0
        images = gpuArray(images) ;
    end
    
    if onlyInput
        inputs = { 'input', images } ;
        return;
    end

    if isfield( ImDb.images, 'loss') 
        loss = ImDb.images.loss(batch) ;
        predY = ImDb.images.predY(batch) ;
        inputs = { 'input', images, 'loss', loss, 'predY', predY } ;
    else
        inputs = { 'input', images } ;
    end

end