function Y = conf_predict(X, Model)
% Y = conf_predict(X, Model)
  
    switch Model.Conf.feat
        case 'linear'
            Z = [X; ones(1,size(X,2))];
            
        case 'quad'
            Z = [qmap(X); ones(1,size(X,2))];
            
        case 'quadnorm'
            Z = qmap(X);
            Z = [Z./repmat(sqrt(sum( Z.^2,1 )),size(Z,1),1 ); ones(1,size(X,2))];
            
        case 'rbf'
            K = kernel( Model.Conf.SV, X, 'rbf', Model.Conf.scale );
            Z = [Model.Conf.A*K; ones(1,size(K,2))];
    end
  
    predY = linclassif( X, Model );
    
    N     = numel( predY );
    conf  = zeros( N, 1);
    for i = 1 : N
        xx = zeros(size(Z,1), Model.nY);
        xx(:, predY(i)) = Z(:,i);
        conf(i) = Model.Conf.W'*xx(:);
    end

    Y = predY;
    Y(find( conf >= Model.Conf.th)) = Model.nY+1;
end