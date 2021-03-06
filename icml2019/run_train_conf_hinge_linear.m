function run_train_conf_hinge_linear( dataSet, setting )

    if nargin < 1
        dataSet = 'codrna1';
        setting = 'msvmlin+hinge1+zmuv+par5';
    end

    switch setting
        
        case 'lr+hinge1+zmuv+par5';

            lambdaRange = [1 10 100 1000];
            Data        = load( ['../data/' dataSet '.mat'], 'X','Y','Split' );
            rootFolder  = ['results/lr/' dataSet '/'];

            Opt.verb    = 1;   
            Opt.tolRel  = 0.01;
            riskType    = 1;
            zmuvNorm    = 1;
            nThreads    = 5;

        case 'msvmlin+hinge1+zmuv+par5';

            lambdaRange = [1 10 100 1000];
            Data        = load( ['../data/' dataSet '.mat'], 'X','Y','Split' );
            rootFolder  = ['results/msvmlin/' dataSet '/'];

            Opt.verb    = 1;   
            Opt.tolRel  = 0.01;
            riskType    = 1;
            zmuvNorm    = 1;
            nThreads    = 5;
    end

    %%
    outFolder = sprintf('%s/conf_hinge%d_linear_zmuv%d_th%d/', rootFolder, riskType, zmuvNorm, nThreads );
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

        for lambda = lambdaRange
            fprintf('[split=%d, lambda=%f]\n', split, lambda );

            modelFile = sprintf('%smodel_split%d_lam%f.mat', outFolder, split, lambda );
            lockFile  = sprintf('%smodel_split%d_lam%f.lock', outFolder, split, lambda );
            
            if ~exist( modelFile ) & ~exist( lockFile )

                fid = fopen( lockFile, 'w+');
                fprintf( fid, '%s\n%s\n', hostname, datestr(now) );            
                fclose(fid);

                if isempty( trnX )
                    if zmuvNorm
                        T    = zmuv( Data.X(:,Data.Split(split).trn2) );
                        trnX = [affinemap(Data.X(:,Data.Split(split).trn2),T); ones(1,nTrn)];
                        valX = [affinemap( Data.X(:,Data.Split(split).val2),T); ones(1,nVal)];
                    else
                        T = [];
                        trnX = [ Data.X(:,Data.Split(split).trn2); ones(1,nTrn)];
                        valX = [ Data.X(:,Data.Split(split).val2); ones(1,nVal)];
                    end
                    trnY = Data.Y(:,Data.Split(split).trn2);
                    valY = Data.Y(:,Data.Split(split).val2);
                end

                if nThreads == 1
                    RrData = risk_rrank_init(trnX, trnPredY, trnPredLoss, nY);
                else
                    RrData = [];
                    idx    = randperm(nTrn);
                    from   = 1;
                    for p = 1 : nThreads
                        to        = round( p*nTrn/nThreads );
                        RrData{p} = risk_rrank_init(trnX(:,idx(from:to)), trnPredY(idx(from:to)), trnPredLoss(idx(from:to)), nY);
                        from      = to + 1;
                    end
                end
                
                % run solver
                if lambda ~= 0
                    switch riskType
                        case 1
                            if nThreads == 1
                                [W, Stat] = bmrm( RrData, @risk_rrank, lambda, Opt );
                            else
%                                [W, Stat] = parbmrm( RrData, @risk_rrank, lambda, Opt );
                                [W, Stat] = bmrm( RrData, @risk_rrank_par, lambda, Opt );
                                
                            end
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
                valAuc  = sum( cumsum( valPredLoss(idx) ))/(nVal^2);

                RrData  = risk_rrank_init(trnX, trnPredY, trnPredLoss, nY);
                conf    = W'*RrData.X;
                [~,idx] = sort( conf );
                trnAuc  = sum( cumsum( RrData.risk(idx)))/(nTrn^2);

                fprintf('trnAuc=%.4f, valAuc =%.4f\n', trnAuc, valAuc);

                save( modelFile, 'W', 'trnAuc', 'valAuc', 'T' );
                
                delete( lockFile );
            end
        end    
    end
    
    %%
    numDone    = 0;
    numMissing = 0;
    for split = 1 : nSplits
        for lambda = lambdaRange
            modelFile = sprintf('%smodel_split%d_lam%f.mat', outFolder, split, lambda );
            if ~exist( modelFile )
                numMissing = numMissing + 1;
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
    trnAuc = zeros( numel( lambdaRange ), nSplits );
    valAuc = zeros( numel( lambdaRange ), nSplits );
    for lambda = lambdaRange
        iLambda = find( lambda == lambdaRange );
        for split = 1 : nSplits
            modelFile = sprintf('%smodel_split%d_lam%f.mat', outFolder, split, lambda );        
            R = load( modelFile, 'trnAuc', 'valAuc' );

            trnAuc( iLambda, split) = R.trnAuc;
            valAuc( iLambda, split) = R.valAuc;
        end
    end

    %% Find best lambda
    fprintf('split   bestLambda   trnerr   valerr\n');
    bestLambda   = nan*ones(nSplits,1);
    for split = 1 : nSplits
        [~,idx ] = min( valAuc(:,split));
        bestLambda(split) = lambdaRange( idx );
        fprintf('%d        %.f           %.4f   %.4f\n', split, bestLambda(split), trnAuc(idx,split),valAuc(idx,split));
    end

    fprintf('    lambda  trnerr          valerr\n');
    for lambda = lambdaRange
        iLambda = find( lambda == lambdaRange );
        fprintf('%10.4f  %.4f(%.4f)  %.4f(%.4f)\n', lambda, ...
            mean( trnAuc(iLambda,:)), std( trnAuc(iLambda,:)), ...
            mean( valAuc(iLambda,:)), std( valAuc(iLambda,:)) );
    end

    %% Evaluate best model on test data
    for split = 1 : nSplits

        predFile   = sprintf('%sresults_split%d.mat', rootFolder, split);         
        modelFile  = sprintf('%smodel_split%d_lam%f.mat', outFolder, split, bestLambda(split) );
        resultFile = sprintf('%sresults_split%d.mat', outFolder, split );

        Pred     = load( predFile );
        predLoss = Pred.predLoss;
        predY    = Pred.predY;

        tstIdx = Data.Split(split).tst;
        nTst   = numel( tstIdx );
        valIdx = Data.Split(split).val2;
        nVal   = numel( valIdx );
        nExamples = size( Data.X, 2);


        if ~exist( resultFile )       

            load( modelFile, 'W', 'T' );

            if zmuvNorm
                X = [affinemap( Data.X, T); ones(1,nExamples)];
            else
                X = [Data.X; ones(1,nExamples)];
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

            valPredLoss    = predLoss( valIdx);
            [~,idx]        = sort( uncertainty( valIdx) );
            valLoss        = sum( cumsum( valPredLoss(idx) ))/(nVal^2);

            save( resultFile, 'tstAuc', 'tstRiskCurve', 'valLoss', 'tstLoss' );
            fprintf( 'results saved to: %s\n', resultFile);
        end
    end

    %% Conmpute AUC and RC-curve
    tstAuc      = [];
    tstLoss     = [];
    valLoss     = [];
    tstRiskCurve = [];
    for split = 1 : nSplits
        resultFile = sprintf('%sresults_split%d.mat', outFolder, split );
        R          = load( resultFile, 'tstAuc', 'tstRiskCurve','valLoss', 'tstLoss' );

        tstAuc       = [tstAuc R.tstAuc ];
        tstLoss      = [tstLoss R.tstLoss];
        valLoss      = [valLoss R.valLoss ];
        tstRiskCurve = [tstRiskCurve R.tstRiskCurve];
    end

    fprintf('tstAuc=%.4f(%.4f), valLoss=%.4f(%.4f)\n', ...
        mean( tstAuc), std( tstAuc ), mean( valLoss), std( valLoss ));

    outFile = [outFolder 'results.mat'];
    save( outFile, 'tstRiskCurve', 'tstAuc', 'valLoss', 'tstLoss' );

    %%
%     figure;
%     plot(  [1:nTst]/nTst, tstRiskCurve );
%     hold on;
%     plot( [1:nTst]/nTst, mean( tstRiskCurve, 2), 'r', 'linewidth', 2);
%     xlabel('cover');
%     ylabel('err');
%     grid on;

    return;
end

