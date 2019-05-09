function inputs = getDagNNBatch(opts, imdb, batch)
% inputs = getDagNNBatch(opts, imdb, batch)
    images = imdb.images.data(:,:,:,batch) ;
    labels = single(imdb.images.labels(1,batch) );
    if rand > 0.5, images=fliplr(images) ; end
    if ~isempty(opts.gpus)
        images = gpuArray(images) ;
    end
    inputs = {'input', images, 'label', labels} ;
end