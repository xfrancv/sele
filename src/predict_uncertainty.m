function uncertainty = predict_uncertainty( X, predY, Model )
%
% uncetainty = predict_uncertainty( X, predY, Model )
%


    nY = numel( Model.W0 );
    N  = size( X, 2);
    uncertainty = zeros( N, 1);
    
    switch Model.type 
        
        case 'linear'
            for y = 1 : nY
                idx = find( predY == y );
                uncertainty( idx ) = ...
                 Model.W(:,y)'*X(:,idx) + repmat(Model.W0(y),1,numel(idx));
            end
end
