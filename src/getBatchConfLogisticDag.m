function inputs = getBatchLogisticDag(Opts,ImDb, batch)
%% 

    images = single( ImDb.images.data(:,:,:,batch) );

    if ~isempty(Opts.gpus) > 0
        images = gpuArray(images) ;
    end
        
    if isfield( ImDb.images, 'loss') 
        loss = ImDb.images.loss(batch) ;
        predY = ImDb.images.predY(batch) ;
        inputs = { 'input', images, 'loss', loss, 'predY', predY } ;
    else
        inputs = { 'input', images } ;
    end

end