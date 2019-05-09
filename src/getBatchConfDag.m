function inputs = getBatchConfDag(Opts, ImDb, batch, onlyInput )
%% 
    if nargin < 4
        onlyInput = 0;
    end

    data  = single( ImDb.images.data(:,:,:,batch) );
    if ~isempty(Opts.gpus) > 0
        data = gpuArray(data) ;
    end
    
    if onlyInput 
        inputs = { 'input', data } ;
        return;
    end
    
    if isfield(ImDb.images,'risk')
        predY = ImDb.images.predY(batch);
        risk  = ImDb.images.risk(batch);
        inputs = { 'input', data, 'risk', risk, 'predY', predY } ;
    else
        inputs = { 'input', data } ;
    end
    
    

end