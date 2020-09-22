function run_train_svorimc( dataSet, setting )

    if nargin < 1
        dataSet = 'california1';
        setting = 'zmuv';
    end
    
    switch setting
        case 'zmuv'
            lambdaRange = [1 0.1 0.01 0.001 0];
            Data = load( ['data/' dataSet '.mat'], 'X','Y','Split' );

            Opt.tolRel = 1e-2;
            Opt.bufSize = 1000;
            Opt.boxConstr = 100;
            zmuvNorm = 1;
            lossType = 'mae';
    end



    %%
    outFolder = ['results/svorimc/' dataSet '/'];
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
                
                Trn = risk_svorimc_init( trnX, trnY);
                
                
                if lambda == 0
                    fprintf('Algorithm: ACCPM');
                    [W,Stat] = accpm( Trn, @risk_svorimc,[],[],Opt.boxConstr, 0, Opt );
                else
                    fprintf('Algorithm: BMRM');
                    W = bmrm( Trn, @risk_svorimc, lambda, Opt );
                end
                
                Model = risk_svorimc_model( Trn, W );
                
                            
                predY = linclassif( trnX, Model);
                trnMae = mean( abs( predY(:)-trnY(:) ));
                trnClsErr = mean( abs( predY(:)~=trnY(:) ));

                predY = linclassif( valX, Model);
                valMae = mean( abs( predY(:)-valY(:) ));
                valClsErr = mean( abs( predY(:)~=valY(:) ));
                
                switch lossType
                    case 'mae'
                        trnErr = trnMae;
                        valErr = valMae;
                    case 'clserr'
                        trnErr = trnClsErr;
                        valErr = valClsErr;
                end

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
        trnIdx = Data.Split(split).trn1;
        nTst   = numel( tstIdx );
        nVal   = numel( valIdx );

        if ~exist( resultFile )       

            load( modelFile, 'Model','T' );
            
            if zmuvNorm
                X = affinemap( Data.X, T);
            else
                X = Data.X;
            end

            
            predY = linclassif( X, Model);
            predLoss = compute_loss( Data.Y, predY, Loss  );
            
            trnErr =mean( predLoss( trnIdx ));
            tstErr =mean( predLoss( tstIdx ));
            valErr =mean( predLoss( valIdx ));
                        
            %
            predY          = predY(:);
            lambda         = bestLambda(split);
            
            save( resultFile, 'tstErr', 'trnErr','valErr', 'lambda', ...
                'predLoss', 'predY', 'Loss' );
            fprintf( 'results saved to: %s\n', resultFile);
        end
    end

    %%
    tstErr = [];
    trnErr = [];
    valErr = [];
    
    for split = 1 : nSplits
        resultFile = sprintf('%sresults_split%d.mat', outFolder, split );
        R          = load( resultFile, 'tstErr', 'valErr', 'trnErr' );

        tstErr       = [tstErr R.tstErr ];
        valErr       = [valErr R.valErr ];
        trnErr       = [trnErr R.trnErr ];
    end

    fprintf('tstErr=%.4f(%.4f), trnErr=%.4f(%.4f), valErr%.4f(%.4f)\n', ...
        mean( tstErr), std( tstErr ), mean(trnErr), std(trnErr),mean(valErr), std(valErr) );

    outFile = [outFolder 'results.mat'];
    save( outFile, 'tstErr', 'valErr', 'tstErr', 'lossType' );

    %%
%     figure;
%     plot(  [1:nTst]/nTst,tstRiskCurve );
%     hold on;
%     plot( [1:nTst]/nTst, mean( tstRiskCurve, 2), 'r', 'linewidth', 2);
%     xlabel('cover');
%     ylabel('err');
%     grid on;

end

