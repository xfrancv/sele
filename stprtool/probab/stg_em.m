function [covIntra,covExtra,Stat] = stg_em( X, Y, Opt )
% STG_EM Estimates parameters of sum of two Gaussioans.
%
% Synopsis:
%  [covIntra,covExtra,Stat] = stg_em( X, Y, Opt )
% 
% Description:
%  Assume the data are generated according to
%    x = mu + a   
%  where mu ~ Gauss(0, covExtra) and a ~ Gauss(0, covIntra).
%  Given sample (X,Y), where X(:,idx) with idx=find(Y==y) denote 
%  data with the same (but unknown) mu, the task is to estimate the
%  covariances (covIntra,covExtra). This code implements the EM
%  algorithm from 
%    D.Chen et al: bayesian Face Revisited: A joint Formulation.
%
% Input:
%   X [nDims x nExamples] the inputs.
%   Y [nExamples x 1] 1..nY definiting the clustering of the inputs in X.
%   Opt [struct]
%     .maxIter [1x1] maximal number of iterations (default 1e2)
%     .eps     [1x1] minimal change in parameters (default 1e-6)
%
% Output:
%  covIntra [nDims x nDims]
%  covExtra [nDims x nDims]
%  


    if nargin < 3, Opt = []; end
    if ~isfield( Opt, 'maxIter'), Opt.maxIter = 1e2; end
    if ~isfield( Opt, 'eps'),     Opt.eps     = 1e-6; end
    if ~isfield( Opt, 'verb'),    Opt.verb    = 1; end

    %%
    [nDims,nExamples] = size(X);
    nY                = max(Y);

    %% center data
    globalMean = mean( X, 2);
    X = X - repmat( globalMean, 1, nExamples );

    %% initial estimate
    classMeans = zeros(nDims, nY );
    covIntra   = zeros(nDims, nDims );
    for y = 1 : nY
        idx             = find( Y == y );
        classMeans(:,y) = mean( X(:, idx), 2);
        tmpX            = X(:,idx) - repmat( classMeans(:,y), 1, length(idx));
        covIntra        = covIntra + cov( tmpX',1 );
    end
    covExtra = cov( classMeans',1 );
    covIntra = covIntra / nY;


    %%
    Stat.eps   = [inf];
    Stat.nIter = 0;
    while Opt.maxIter > Stat.nIter & Stat.eps(end) > Opt.eps

        Stat.nIter = Stat.nIter + 1;

        Xintra     = zeros(nDims, nExamples);
        classMeans = zeros(nDims, nY); 
        F = inv( covIntra );

        for y = 1 : nY
            idx = find( Y == y);
            m   = length(idx);
            G = -inv(m*covExtra+covIntra)*covExtra*F;

            tmpX = sum(X(:,idx),2);
            classMeans(:,y) = covExtra*(F+m*G)*tmpX;
            for i = idx(:)'
                Xintra(:,i) = X(:,i) + covIntra*G*tmpX;
            end
        end

        oldCovExtra = covExtra;
        oldCovIntra = covIntra;
        covExtra    = cov( classMeans',1 );
        covIntra    = cov( Xintra',1 );

        Stat.eps(Stat.nIter) = sum(sum( (oldCovExtra - covExtra).^2)) + ...
                                 sum(sum( ( oldCovIntra-covIntra).^2)); 

        if Opt.verb, fprintf('%2d: |oldParam-newParam|^2 = %d\n', Stat.nIter, Stat.eps(end)); end
    end

end

