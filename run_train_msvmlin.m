function run_train_msvmlin( dataSet, setting )

    if nargin < 1
        dataSet = 'covtype1';
        setting = 'zmuv+reg0.1-100';
    end
    
    switch setting

        case 'zmuv+reg0.1-100'

            Data = load( ['data/' dataSet '.mat'], 'X','Y','Split' );

            Params = [];
            for C = [0.1 1 10 100]
                Params(end+1).C = C;
            end
            zmuvNorm = 1;
            lossType = 'clserr';
            Opts.tolRel = 0.01;
            Opts.bufSize = 5000;
            Opts.verb = 1;
    end
            

    %%
    outFolder = ['results/msvmlin/' dataSet '/'];
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

        for p = 1 : numel( Params )
            fprintf('[split=%d, param=%s]\n', split, svm_param_str(Params(p)) );

            modelFile   = sprintf('%smodel_split%d_param_%s.mat', outFolder, split, svm_param_str(Params(p)) ); 
            modelFolder = sprintf('%smodel_split%d_param_%s/', outFolder, split, svm_param_str(Params(p)) );
            lockFile    = sprintf('%smodel_split%d_param_%s.lock', outFolder, split, svm_param_str(Params(p)) );
            
            if ~exist( modelFile ) & ~exist( lockFile )

                fid = fopen( lockFile, 'w+');
                fprintf( fid, '%s\n%s\n', hostname, datestr(now) );            
                fclose(fid);
                
                nTrn = numel( Data.Split(split).trn1);
                nVal = numel( Data.Split(split).val1);

                if zmuvNorm
                    T    = zmuv( Data.X(:,Data.Split(split).trn1) );
                    trnX = [affinemap(Data.X(:,Data.Split(split).trn1),T); ones(1,nTrn)];
                    valX = [affinemap( Data.X(:,Data.Split(split).val1),T); ones(1,nVal)];
                else
                    T    = [];
                    trnX = [Data.X(:,Data.Split(split).trn1); ones(1,nTrn)];
                    valX = [Data.X(:,Data.Split(split).val1); ones(1,nVal)];
                end
                trnY = Data.Y(:,Data.Split(split).trn1);
                valY = Data.Y(:,Data.Split(split).val1);

                %
%                Svm = msvmb2( trnX, trnY, Params(p).C, 'rbf', Params(p).rbfWidth, Opt );
                if nY > 2
                    
                    W = msvmocas( trnX, trnY, Params(p).C, 1,Opts.tolRel, 0, 0, Opts.bufSize, inf, inf, Opts.verb);
                    [~,trnPredY] = max(W'*trnX);
                    [~,valPredY] = max(W'*valX);
                else
                    bintrnY = trnY;
                    bintrnY(find(bintrnY==2)) = -1;
                    W = svmocas( trnX,0,bintrnY,Params(p).C, 1,Opts.tolRel,0,0,Opts.bufSize,inf,inf,Opts.verb);

                    trnPredY = sign2(W'*trnX);
                    trnPredY(find(trnPredY==-1)) = 2;
                    valPredY = sign2(W'*valX);
                    valPredY(find(valPredY==-1)) = 2;
                end

                trnErr = mean( compute_loss( trnY, trnPredY, Loss));
                valErr = mean( compute_loss( valY, valPredY, Loss));
                
                fprintf('trnerr=%.4f, valerr=%.4f\n', trnErr, valErr);

                save( modelFile, 'W', 'trnErr', 'valErr', 'lossType','T'  );

                delete( lockFile );

            end
        end    
    end

    %%
    numDone    = 0;
    numMissing = 0;
    for split = 1 : nSplits
        for p = 1 : numel( Params )
            modelFile   = sprintf('%smodel_split%d_param_%s.mat', outFolder, split, svm_param_str(Params(p)) ); 
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
    trnErr = zeros( numel( Params ), nSplits );
    valErr = zeros( numel( Params ), nSplits );
    for p = 1 : numel( Params )
        for split = 1 : nSplits
            modelFile   = sprintf('%smodel_split%d_param_%s.mat', outFolder, split, svm_param_str(Params(p)) ); 
            R = load( modelFile, 'trnErr', 'valErr' );

            trnErr( p, split) = R.trnErr;
            valErr( p, split) = R.valErr;
        end

    end

    fprintf('split param                                trnerr   valerr\n');
    bestParams   = nan*ones(nSplits,1);
    for split = 1 : nSplits
        [~,idx ] = min( valErr(:,split));
        bestParams(split) = idx;
        fprintf('%d    %2d   %30s   %.4f   %.4f\n', split, idx, svm_param_str(Params(idx)), trnErr(idx,split),valErr(idx,split));
    end

    fprintf(' param                              trnerr          valerr\n');
    for p = 1 : numel( Params )
        fprintf('%2d  %30s  %.4f(%.4f)  %.4f(%.4f)\n',p,svm_param_str(Params(p)), ...
            mean( trnErr(p,:)), std( trnErr(p,:)), ...
            mean( valErr(p,:)), std( valErr(p,:)) );
    end

    %%
    for split = 1 : nSplits
        modelFile   = sprintf('%smodel_split%d_param_%s.mat', outFolder, split, svm_param_str(Params(bestParams(split)) )); 
        resultFile = sprintf('%sresults_split%d.mat', outFolder, split );

        tstIdx = Data.Split(split).tst;
        nTst   = numel( tstIdx );
        valIdx = Data.Split(split).val1;
        nVal   = numel( valIdx );
        
        nDims     = size(Data.X,1);
        nY        = max( Data.Y );
        nExamples = size(Data.X,2);
        
        if ~exist( resultFile )       

            load( modelFile, 'W', 'T' );

            
            if zmuvNorm
                X = [affinemap( Data.X, T); ones(1,nExamples)];
            else
                X = [Data.X; ones(1,nExamples)];
            end

            if size( W,2) > 1
                % multiclass
                [predScore,predY] = max( W'*X);
                rscore = -predScore(:);
            else
                % two-classes
                predScore  = W'*X;
                predY = sign2(predScore);
                predY(find(predY==-1)) = 2;
                rscore = -abs(predScore(:));
            end
                
            predLoss      = compute_loss( Data.Y, predY, Loss );
            tstErr        = mean( predLoss( tstIdx ) );

            tstPredLoss   = predLoss( tstIdx );
            [~,idx]       = sort( rscore( tstIdx ) );
            tstRiskCurve  = cumsum( tstPredLoss(idx))./[1:nTst]';
            tstAuc        = mean( tstRiskCurve );
            tstLoss       = sum( cumsum( tstPredLoss(idx) ))/(nTst^2);

            valPredLoss   = predLoss( valIdx);
            [~,idx]       = sort( rscore( valIdx) );
            valLoss       = sum( cumsum( valPredLoss(idx) ))/(nVal^2);
            
            %
            rscore        = rscore(:);
            predY         = predY(:);

            save( resultFile, 'predScore','rscore','predY', 'tstErr', ...
                'tstRiskCurve',  'predLoss', 'tstAuc','valLoss','tstLoss', 'Loss' );
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
        R          = load( resultFile, 'tstErr', 'tstRiskCurve', 'tstAuc','tstLoss', 'valLoss' );

        tstErr       = [tstErr R.tstErr ];
        valLoss      = [valLoss R.valLoss ];
        tstAuc       = [tstAuc R.tstAuc ];        
        tstLoss      = [tstLoss R.tstLoss ];    
        tstRiskCurve = [tstRiskCurve R.tstRiskCurve];
    end

    fprintf('tstErr=%.4f(%.4f), tstAuc=%.4f(%.4f), valLoss=%.4f(%.4f)\n', ...
        mean( tstErr), std( tstErr ), mean(tstAuc), std(tstAuc),mean(valLoss), std(valLoss) );

    outFile = [outFolder 'results.mat'];
    save( outFile, 'tstRiskCurve', 'tstErr', 'tstAuc', 'valLoss','tstLoss', 'lossType' );

    %%
%     figure;
%     plot(  [1:nTst]/nTst,tstRiskCurve );
%     hold on;
%     plot( [1:nTst]/nTst, mean( tstRiskCurve, 2), 'r', 'linewidth', 2);
%     xlabel('cover');
%     ylabel('err');
%     grid on;

end
