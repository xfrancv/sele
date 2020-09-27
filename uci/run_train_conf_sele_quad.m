function run_train_conf_sele_quad( dataSet, setting, trnData )

    if nargin < 1 
        dataSet = 'avila1';
        setting = 'lr+hinge1+zmuv';
    end
    
    switch setting
        case 'lr+hinge1+zmuv';

            Data        = load( ['../data/' dataSet '.mat'], 'X','Y','Split' );
            rootFolder  = ['results/lr/' dataSet '/'];

            Params = [];
            for lambda = [1 10 100 1000 ]
                for batchSize = [50 100 500 1000]
                    Params(end+1).lambda = lambda;
                    Params(end).batchSize = batchSize;
                end
            end
            
            
            Opt.verb    = 1;   
            Opt.tolRel  = 0.01;
            riskType    = 1;
            zmuvNorm    = 1;

        case 'msvmlin+hinge1+zmuv';

            Data        = load( ['../data/' dataSet '.mat'], 'X','Y','Split' );
            rootFolder  = ['results/msvmlin/' dataSet '/'];
            
            Params = [];
            for lambda = [1 10 100 1000 ]
                for batchSize = [50 100 500 1000]
                    Params(end+1).lambda = lambda;
                    Params(end).batchSize = batchSize;
                end
            end

            Opt.verb    = 1;   
            Opt.tolRel  = 0.01;
            riskType    = 1;
            zmuvNorm    = 1;
    end

    %%
    if nargin >= 3
        Data = take_trn2_data( Data, trnData );
        outFolder = sprintf('%s/conf_sele%d_quad_zmuv%d_trn%.f/', rootFolder, riskType, zmuvNorm, trnData );
    else
        outFolder = sprintf('%s/conf_sele%d_quad_zmuv%d/', rootFolder, riskType, zmuvNorm );
    end
       
    if ~exist( outFolder ), mkdir( outFolder ); end


    %%
    nSplits = numel( Data.Split );
    N       = size( Data.X, 2);
    nY      = max( Data.Y );

    for split = 1 : nSplits

        trnX = [];

        predFile    = sprintf('%sresults_split%d.mat', rootFolder, split);        
        Pred        = load( predFile );
        trnPredY    = Pred.predY( Data.Split(split).trn2 );
        valPredY    = Pred.predY( Data.Split(split).val2 );
        trnPredLoss = Pred.predLoss( Data.Split(split).trn2 );
        valPredLoss = Pred.predLoss( Data.Split(split).val2 );
        nTrn        = numel( Data.Split(split).trn2 );
        nVal        = numel( Data.Split(split).val2 );
    
        for p = 1 : numel( Params )

            batchSize = Params(p).batchSize;
            lambda = Params(p).lambda;

            fprintf('[split=%d, params=%s]\n', split, sele_param_str(Params(p)) );

            modelFile = sprintf('%smodel_split%d_param_%s.mat', outFolder, split, sele_param_str(Params(p)) );
            lockFile  = sprintf('%smodel_split%d_param_%s.lock', outFolder, split, sele_param_str(Params(p)) );

            if ~exist( modelFile ) & ~exist( lockFile )
                
                fid = fopen( lockFile, 'w+');
                fprintf( fid, '%s\n%s\n', hostname, datestr(now) );            
                fclose(fid);                

                if isempty( trnX )
                    if zmuvNorm
                        T    = zmuv( Data.X(:,Data.Split(split).trn2) );
                        trnX = [qmap( affinemap(Data.X(:,Data.Split(split).trn2),T)); ones(1,nTrn)];
                        valX = [qmap( affinemap( Data.X(:,Data.Split(split).val2),T)); ones(1,nVal)];
                    else
                        T = [];
                        trnX = [qmap( Data.X(:,Data.Split(split).trn2)); ones(1,nTrn)];
                        valX = [ qmap(Data.X(:,Data.Split(split).val2)); ones(1,nVal)];
                    end
                    trnY = Data.Y(:,Data.Split(split).trn2);
                    valY = Data.Y(:,Data.Split(split).val2);
                end

                nBatches = max(1,round(nTrn/batchSize));
                RrData = [];
                idx    = randperm(nTrn);
                from   = 1;
                for p = 1 : nBatches
                    to        = round( p*nTrn/nBatches );
                    RrData{p} = risk_rrank_init(trnX(:,idx(from:to)), trnPredY(idx(from:to)), trnPredLoss(idx(from:to)), nY);
                    from      = to + 1;
                end
                
                % run solver
                if lambda ~= 0
                    switch riskType
                        case 1
                            [W, Stat] = bmrm( RrData, @risk_rrank_par, lambda, Opt );
                        case 2
                            [W, Stat] = bmrm( RrData, @risk_rrank2, lambda, Opt );
                    end
                else
                    boxConstr = ones( size(RrData.X,1),1)*1000;
                    switch riskType
                        case 1
                            [W, Stat] = accpm( RrData, @risk_rrank,[],[],boxConstr, lambda, Opt);
                        case 2
                            [W, Stat] = accpm( RrData, @risk_rrank2,[],[],boxConstr, lambda, Opt);
                    end
                end

                conf  = zeros( nVal, 1);
                for i = 1 : nVal
                    xx                 = zeros(size(valX,1),nY);
                    xx(:, valPredY(i)) = valX(:,i);
                    conf(i)            = W'*xx(:);
                end

                
                [~,idx] = sort( conf );
                valRiskCurve = cumsum( valPredLoss(idx))./[1:nVal]';
                valAuc  = mean( valRiskCurve );                
                valLoss  = sum( cumsum( valPredLoss(idx) ))/(nVal^2);
                
                RrData  = risk_rrank_init(trnX, trnPredY, trnPredLoss, nY);
                conf    = W'*RrData.X;
                [~,idx] = sort( conf );
                trnRiskCurve = cumsum( RrData.risk(idx))./[1:nTrn]';
                trnAuc  = mean( trnRiskCurve );
                trnLoss  = sum( cumsum( trnPredLoss(idx) ))/(nTrn^2);

                fprintf('trnAuc=%.4f, valAuc =%.4f, trnLoss=%.4f, valLoss=%.4f\n', trnAuc, valAuc, trnLoss, valLoss);
                
                save( modelFile, 'W', 'trnLoss', 'valLoss', 'T', 'trnAuc','valAuc' );
                
                delete( lockFile );
            end
        end    
    end

    %%
    numDone    = 0;
    numMissing = 0;
    for split = 1 : nSplits
        for p = 1 : numel( Params)
            modelFile = sprintf('%smodel_split%d_param_%s.mat', outFolder, split, sele_param_str(Params(p)) );
            if ~exist( modelFile )
                numMissing = numMissing + 1;
                fprintf('missing: %s\n', modelFile);                
            else
                numDone  = numDone + 1;
            end
        end
    end
    fprintf('#done=%d\n', numDone);
    fprintf('#missing=%d\n', numMissing );
    if numMissing
        return;
    end    
    
    
    %% Collect results
    trnAuc = zeros( numel( Params ), nSplits );
    valAuc = zeros( numel( Params ), nSplits );
    for p = 1 : numel(  Params )
        for split = 1 : nSplits
            modelFile = sprintf('%smodel_split%d_param_%s.mat', outFolder, split, sele_param_str(Params(p)) );
            R = load( modelFile, 'trnAuc', 'valAuc' );

            trnAuc( p, split) = R.trnAuc;
            valAuc( p, split) = R.valAuc;
        end
    end

    %% Find best lambda
    fprintf('split   param                                          trnAuc   valAuc\n');
    bestParams   = nan*ones(nSplits,1);
    for split = 1 : nSplits
        [~,idx ] = min( valAuc(:,split));
        bestParams(split) = idx;
        fprintf('%d        %2d   %30s           %.4f   %.4f\n', split, idx, ...
            sele_param_str(Params(idx)), trnAuc(idx,split), valAuc(idx,split));
    end

    fprintf('param                                        trnAuc          valAuc\n');
    for p = 1 : numel( Params )
        fprintf('%2d   %30s  %.4f(%.4f)  %.4f(%.4f)\n', p, sele_param_str(Params(p)), ...
            mean( trnAuc(p,:)), std( trnAuc(p,:)), ...
            mean( valAuc(p,:)), std( valAuc(p,:)) );
    end

    %% Evaluate best model on test data
    for split = 1 : nSplits

        predFile   = sprintf('%sresults_split%d.mat', rootFolder, split);         
        modelFile = sprintf('%smodel_split%d_param_%s.mat', outFolder, split, sele_param_str(Params(bestParams(split)) ));
        resultFile = sprintf('%sresults_split%d.mat', outFolder, split );

        Pred        = load( predFile );
        predLoss = Pred.predLoss;
        predY    = Pred.predY;

        tstIdx = Data.Split(split).tst;
        nTst   = numel( tstIdx );
        valIdx = Data.Split(split).val2;
        nVal   = numel( valIdx );
        nExamples = size( Data.X,2);


        if ~exist( resultFile )       

            load( modelFile, 'W', 'T' );

            if zmuvNorm
                X = [qmap( affinemap( Data.X, T)); ones(1,nExamples)];
            else
                X = [qmap(Data.X); ones(1,nExamples)];
            end
            
            uncertainty = zeros( nExamples,1);
            for i = 1 : nExamples
                xx = zeros(size( X,1),nY);
                xx(:, predY(i)) = X(:,i);
                uncertainty(i) = W'*xx(:);
            end
            
            tstPredLoss   = predLoss( tstIdx );
            [~,idx]       = sort( uncertainty( tstIdx ) );
            tstRiskCurve  = cumsum( tstPredLoss(idx))./[1:nTst]';
            tstAuc        = mean( tstRiskCurve);
            tstLoss       = sum( cumsum( tstPredLoss(idx) ))/(nTst^2);

            valPredLoss   = predLoss( valIdx);
            [~,idx]       = sort( uncertainty( valIdx) );
            valLoss       = sum( cumsum( valPredLoss(idx) ))/(nVal^2);
            
            save( resultFile, 'tstAuc', 'tstLoss', 'valLoss','tstRiskCurve' );
            fprintf( 'results saved to: %s\n', resultFile);
        end
    end

    %% Conmpute AUC and RC-curve
    tstAuc       = [];
    tstLoss      = [];
    valLoss      = [];
    tstRiskCurve = [];
    for split = 1 : nSplits
        resultFile = sprintf('%sresults_split%d.mat', outFolder, split );
        R          = load( resultFile, 'tstAuc', 'tstRiskCurve', 'valLoss', 'tstLoss' );

        tstAuc       = [tstAuc R.tstAuc ];
        tstLoss      = [tstLoss R.tstLoss ];
        valLoss      = [valLoss R.valLoss ];
        tstRiskCurve = [tstRiskCurve R.tstRiskCurve];
    end

    fprintf('tstAuc=%.4f(%.4f), tstLoss=%.4f(%.4f), valLoss=%.4f(%.4f)\n',...
        mean( tstAuc), std( tstAuc ), mean( tstLoss), std( tstLoss ),mean( valLoss), std( valLoss ));

    outFile = [outFolder 'results.mat'];
    save( outFile, 'tstRiskCurve', 'tstAuc', 'tstLoss', 'valLoss' );

    %%
% 
    figure;
    plot(  [1:nTst]/nTst, tstRiskCurve );
    hold on;
    plot( [1:nTst]/nTst, mean( tstRiskCurve, 2), 'r', 'linewidth', 2);
    xlabel('cover');
    ylabel('err');
    grid on;
    
    return;
end


