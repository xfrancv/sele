function inputs = getBatchDag(Opts,ImDb, batch)
%% 

    images = single( ImDb.images.data(:,:,:,batch) );

    if ~isempty(Opts.gpus) > 0
        images = gpuArray(images) ;
    end
        
    if isfield( ImDb.images, 'labels') 
        labels = ImDb.images.labels(batch) ;
        inputs = { 'input', images, 'label', labels } ;
    else
        inputs = { 'input', images } ;
    end

end