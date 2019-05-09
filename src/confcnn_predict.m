function conf = confcnn_predict( ImDb, Net, nY, batchSize, getBatch )
%
%  post = confcnn_predict( data, Net, nY, batchSize )
%

    M     = size( ImDb.images.data, 4 );
    score = zeros( nY, M);  
    tic;    
    for t = 1:batchSize:M
        
        batchSize = min( batchSize, M - t + 1) ;
        
        batchStart = t;
        batchEnd   = min( t + batchSize-1, M );
        batch      = [batchStart : batchEnd ];  

        fprintf('batch %d-%d %.2f%%\n', batchStart, batchEnd, batchEnd*100/M);
        
%         dataBatch  = single( data(:,:,:,batch) );
% 
%         if strcmpi(Net.device,'gpu')
%             dataBatch = gpuArray( dataBatch ) ;
%         end

%        inputs = { 'input', dataBatch } ;
        inputs = getBatch( ImDb, batch);
        inputs = inputs(1:2);
                
        Net.mode = 'test' ;
        Net.eval( inputs ) ;
        
        cscore = gather( Net.vars( getVarIndex( Net, 'prediction' ) ).value );

        score(:,batch) = reshape( cscore, nY, numel(batch) );

%         score = gather( Net.vars( getVarIndex( Net, 'prediction' ) ).value );        
%         post(:,batch) = reshape( score, nY, numel(batch) );
    end    
    runtime  = toc;
    
%     post = exp(post);
%     post = post ./ repmat( sum(post), nY, 1);
 
    conf = zeros(M,1);
    for i = 1 : M
        conf(i) = score( ImDb.images.predY(i),i);
    end
end
