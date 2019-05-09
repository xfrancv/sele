function risk = true_risk_landmark(Data, W )
% TRUE_RISK_FCBMAXSUM
%
%  risk = true_risk_fcbmaxsum(Data, W )
%

    risk = 0;

    for i = 1 : length( Data.idx )

        j = Data.idx( i );

        Q = reshape(W(1:Data.nFeatDims*Data.nLabels),Data.nFeatDims,Data.nLabels);
        G = W(Data.nFeatDims*Data.nLabels+1:end);

        projQ = Q'*Data.X(:,j);

        V = [projQ(:) ; G(:)];

        score    = Data.feat * V;
        [~, idx] = max(score);

        estY  = Data.feat( idx, 1:Data.nLabels );    
        loss  = sum( estY(:) ~= Data.Y(:,j)) / Data.nLabels;
        
        risk  = risk + loss;
          
    end

    risk = risk / length( Data.idx );

end
