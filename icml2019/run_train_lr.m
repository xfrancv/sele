function run_train_mlogreg_linear( dataSet, setting )

    if nargin < 1
        dataSet = 'avila1';
        setting = 'zmuv+reg0-100';
    end
    
    switch setting
        case 'zmuv+reg0-100'
            lambdaRange = [0 1 10 100];
            Data = load( ['data/' dataSet '.mat'], 'X','Y','Split' );

            Opt.maxIter = 100;
            Opt.eps     = 1.e-5;
            Opt.m       = 5;
            Opt.verb    = 1;      
            X0 = 1;
            zmuvNorm = 1;
            lossType = 'clserr';
    end



    %%
    outFolder = ['results/lr/' dataSet '/'];
    if ~exist( outFolder ), mkdir( outFolder ); end

    %%
    nSplits = numel( Data.Split );
    N       = size( Data.X, 2);
    nY      = max( Data.Y );
    
    switch lossType
        case 'mae'
            Loss    = abs( repmat([1:nY],nY,1)-repmat([1:nY]',1,nY) );
        case 'clserr'
            Loss    = ones(nY,nY) - eye(nY,nY);
    end

    for split = 1 : nSplits

        trnX = [];

        for lambda = lambdaRange
            fprintf('[split=%d, lambda=%f]\n', split, lambda );

            modelFile = sprintf('%smodel_split%d_lam%f.mat', outFolder, split, lambda );
            lockFile  = sprintf('%smodel_split%d_lam%f.lock', outFolder, split, lambda );

            if ~exist( modelFile ) & ~exist( lockFile )

                fid = fopen( lockFile, 'w+');
                fprintf( fid, '%s\n%s\n', hostname, datestr(now) );            
                fclose(fid);

                nTrn        = numel( Data.Split(split).trn1 );
                nVal        = numel( Data.Split(split).val1 );
                
                if isempty( trnX )
                    if zmuvNorm
                        T    = zmuv( Data.X(:,Data.Split(split).trn1) );
                        trnX = affinemap( Data.X(:,Data.Split(split).trn1),T); 
                        valX = affinemap( Data.X(:,Data.Split(split).val1),T);
                    else
                        trnX = Data.X(:,Data.Split(split).trn1);
                        valX = Data.X(:,Data.Split(split).val1);
                    end

                    trnY = Data.Y(Data.Split(split).trn1);
                    valY = Data.Y(Data.Split(split).val1);
                end

                nTrn = numel( trnY );
                nVal = numel( valY );
                
                Trn    = risk_mlogreg_init( trnX, X0, trnY, lambda );
                W      = lbfgs( Trn, 'risk_mlogreg', [], Opt );
                Model  = risk_mlogreg_model( Trn, W );     

                post      = exp( Model.W'* trnX + repmat( Model.W0, 1, nTrn ));
                post      = post ./ repmat( sum( post, 1), nY, 1);                
                [~,predY] = min( Loss*post );                
                trnErr    = mean( compute_loss( predY, trnY, Loss ));
                
                post      = exp( Model.W'* valX + repmat( Model.W0, 1, nVal ));
                post      = post ./ repmat( sum( post, 1), nY, 1);                
                [~,predY] = min( Loss*post );                

                valErr = mean( compute_loss( predY, valY, Loss));

                fprintf('trnerr=%.4f, valerr=%.4f\n', trnErr, valErr);

                save( modelFile, 'Model', 'trnErr', 'valErr', 'lossType','T' );

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

    %%
    trnErr = zeros( numel( lambdaRange ), nSplits );
    valErr = zeros( numel( lambdaRange ), nSplits );
    for lambda = lambdaRange
        iLambda = find( lambda == lambdaRange );
        for split = 1 : nSplits
            modelFile = sprintf('%smodel_split%d_lam%f.mat', outFolder, split, lambda );
            R = load( modelFile, 'trnErr', 'valErr' );

            trnErr( iLambda, split) = R.trnErr;
            valErr( iLambda, split) = R.valErr;
        end

    end

    fprintf('split   bestLambda   trnerr   valerr\n');
    bestLambda   = nan*ones(nSplits,1);
    for split = 1 : nSplits
        [~,idx ] = min( valErr(:,split));
        bestLambda(split) = lambdaRange( idx );
        fprintf('%d        %.f           %.4f   %.4f\n', split, bestLambda(split), trnErr(idx,split),valErr(idx,split));
    end


    fprintf('    lambda  trnerr          valerr\n');
    for lambda = lambdaRange
        iLambda = find( lambda == lambdaRange );
        fprintf('%10.4f  %.4f(%.4f)  %.4f(%.4f)\n', lambda, ...
            mean( trnErr(iLambda,:)), std( trnErr(iLambda,:)), ...
            mean( valErr(iLambda,:)), std( valErr(iLambda,:)) );
    end

    %%
    for split = 1 : nSplits
        modelFile  = sprintf('%smodel_split%d_lam%f.mat', outFolder, split, bestLambda(split) );
        resultFile = sprintf('%sresults_split%d.mat', outFolder, split );

        tstIdx = Data.Split(split).tst;
        valIdx = Data.Split(split).val1;
        nTst   = numel( tstIdx );
        nVal   = numel( valIdx );

        if ~exist( resultFile )       

            load( modelFile, 'Model','T' );
            
            if zmuvNorm
                post   = exp( Model.W'* affinemap( Data.X, T) + repmat( Model.W0, 1, N ));
            else
                post   = exp( Model.W'* Data.X + repmat( Model.W0, 1, N ));
            end

            post   = post ./ repmat( sum( post, 1), nY, 1);

            [rscore,predY] = min( Loss * post );

            predLoss       = compute_loss( Data.Y, predY, Loss  );
            tstErr         = mean( predLoss( tstIdx) );

            tstPredLoss    = predLoss( tstIdx);
            [~,idx]        = sort( rscore( tstIdx) );
            tstRiskCurve   = cumsum( tstPredLoss(idx))./[1:nTst]';
            tstAuc         = mean( tstRiskCurve ) ;
            tstLoss        = sum( cumsum( tstPredLoss(idx) ))/(nTst^2);

            valPredLoss    = predLoss( valIdx);
            [~,idx]        = sort( rscore( valIdx) );
            valLoss        = sum( cumsum( valPredLoss(idx) ))/(nVal^2);
            
            %
            rscore         = rscore(:);
            predY          = predY(:);
            lambda         = bestLambda(split);
            
            save( resultFile, 'valLoss','post','rscore', 'predY', 'tstErr', ...
                'tstRiskCurve', 'lambda', 'predLoss', 'tstAuc', 'tstLoss', 'Loss' );
            fprintf( 'results saved to: %s\n', resultFile);
        end
    end

    %%
    tstErr       = [];
    tstAuc       = [];
    tstLoss      = [];
    valLoss      = [];
    tstRiskCurve = [];
    
    for split = 1 : nSplits
        resultFile = sprintf('%sresults_split%d.mat', outFolder, split );
        R          = load( resultFile, 'tstErr', 'tstRiskCurve', 'tstAuc','tstLoss','valLoss' );

        tstRiskCurve = [tstRiskCurve  R.tstRiskCurve];
        tstErr       = [tstErr R.tstErr ];
        tstAuc       = [tstAuc R.tstAuc ];
        tstLoss      = [tstLoss R.tstLoss ];
        valLoss      = [valLoss R.valLoss ];
    end

    fprintf('tstErr=%.4f(%.4f), tstAuc=%.4f(%.4f), valLoss=%.4f(%.4f)\n', ...
        mean( tstErr), std( tstErr ), mean(tstAuc), std(tstAuc),mean(valLoss), std(valLoss) );

    outFile = [outFolder 'results.mat'];
    save( outFile, 'tstRiskCurve', 'tstErr', 'tstAuc', 'valLoss', 'tstLoss', 'lossType' );

    %%
%     figure;
%     plot(  [1:nTst]/nTst,tstRiskCurve );
%     hold on;
%     plot( [1:nTst]/nTst, mean( tstRiskCurve, 2), 'r', 'linewidth', 2);
%     xlabel('cover');
%     ylabel('err');
%     grid on;

end

