function [R,subgrad,Data] = risk_latentsvm( Data, W )
% RISK_LATENTSVM Hinge loss for latent linear classifier.
% 
% Synopsis:
%  [R,subgrad,Data] = risk_latentsvm( Data )
%  [R,subgrad,Data] = risk_latentsvm( Data, W )
%

    nZ        = Data.nZ;
    nDim      = Data.nDim;
    nExamples = Data.nExamples;

    if nargin < 2, W = zeros( nDim*nZ, 1 ); end

    W    = reshape( W, nDim, nZ );
    proj = W'*Data.X + Data.Loss;

    [tmp, predZ ] = max( proj );
    R = tmp*Data.gamma - sum( sum( Data.Psi0.*W ));
    R = R / nExamples;

    subgrad = zeros( nDim, nZ );
    for z = 1 : nZ
        idx          = find( z == predZ );   
        subgrad(:,z) = Data.X(:, idx)*Data.gamma(idx) - Data.Psi0(:,z);   
    end

    subgrad = subgrad(:) / nExamples;

end