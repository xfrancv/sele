function Results = train_msvm( X, X0, Y, Splits, setOfC, outDir, Opt ) 
% TRAIN_MSVM 
%
%   Results = train_msvm( X, X0, Y, Splits, setOfC, outDir, Opt )
%
%  MSVM minimizes
%      F(W) = 0.5*||W||^2 + C/N * sum_{i=1}^N loss_i(W)
%

    if ~isfield( Opt, 'verb' ),    Opt.verb = 1; end
    if ~isfield( Opt, 'zmuv' ),    Opt.zmuv = 1; end
    if ~isfield( Opt, 'bufSize' ), Opt.bufSize = 2000; end
    if ~isfield( Opt, 'tolAbs' ),  Opt.tolAbs  = 0; end
    if ~isfield( Opt, 'tolRel' ),  Opt.tolRel  = 0.01; end

    resultFile = [outDir 'results.mat'];

    nY = max(Y);

    if ~exist( resultFile )

        if ~exist( outDir ), mkdir( outDir ); end
        
        nSplits = length( Splits.trnSplit );

        for split = 1 : nSplits

            trnX = [];

            for C = setOfC(:)'

                modelFile = sprintf('%smodel_split%d_C%f.mat' , outDir, split, C );
                lockFile  = sprintf('%smodel_split%d_C%f.lock', outDir, split, C );

                if ~exist( modelFile ) & ~exist( lockFile )

                    fid = fopen( lockFile, 'w+' );
                    fprintf( fid, 'Training started at %s\n', datestr( now ));
                    fclose( fid );

                    if isempty( trnX )


                        trnX   = X(:,Splits.trnSplit{split} );
                        trnY   = Y(Splits.trnSplit{split} );

                        valX   = X(:,Splits.valSplit{split});
                        valY   = Y(Splits.valSplit{split});

                        tstX   = X(:,Splits.tstSplit{split});
                        tstY   = Y(Splits.tstSplit{split});


                        % zero mean unit variance normalization of features
                        if Opt.zmuv                         
                            NormModel = zmuv( trnX );
                            trnX = affinemap( trnX, NormModel );
                            valX = affinemap( valX, NormModel );
                            tstX = affinemap( tstX, NormModel );                        
                        end

                        % biased rule
                        if X0 
                            trnX = [trnX ; X0*ones(1,size(trnX,2))];
                            valX = [valX ; X0*ones(1,size(valX,2))];
                            tstX = [tstX ; X0*ones(1,size(tstX,2))];
                        end
                    end

                    nExamples = size(trnX, 2);

                    [W,Stat] = msvmocas(trnX,trnY,C/nExamples,1,...
                        Opt.tolRel,Opt.tolAbs,0,Opt.bufSize,inf,inf,Opt.verb );

                    % build linear classifier 
                    W = reshape(W, size(trnX,1), nY );
                    if X0
                        Model.W  = W(1:end-1,:);
                        Model.W0 = W(end,:)';
                    else
                        Model.W  = W;
                        Model.W0 = zeros(nY,1);
                    end

                    % compute errors
                    [~,estY] = max( W'*trnX );
                    trnErr   = sum( estY(:)~=trnY(:) )/ length(trnY );

                    [~,estY] = max( W'*valX );
                    valErr   = sum( estY(:)~=valY(:) )/ length(valY );

                    [~,estY] = max( W'*tstX );
                    tstErr   = sum( estY(:)~=tstY(:) )/ length(tstY );

                    save( modelFile, 'Model','trnErr','valErr','tstErr', 'Opt','C' );                    

                    delete( lockFile );
                end

            end
        end


        missingModel = false;
        for split = 1 : nSplits
            for C = setOfC(:)'
                modelFile = sprintf('%smodel_split%d_C%f.mat' , outDir, split, C );
                if ~exist( modelFile )
                    fprintf('Missing model:\n%s\n', modelFile);
                    missingModel = true;
                end
            end
        end

        if missingModel
            fprintf('Wait until all models are trained.\n');
            Results = [];
            return;
        end

        %% collect results
        trnErrs = zeros( length( setOfC), nSplits );
        tstErrs = zeros( length( setOfC), nSplits );
        valErrs = zeros( length( setOfC), nSplits );
        for split = 1 : nSplits
            for i = 1 : length( setOfC )

                modelFile = sprintf('%smodel_split%d_C%f.mat' , outDir, split, setOfC(i) );

                load( modelFile, 'trnErr','tstErr','valErr');
                trnErrs(i,split) = trnErr;
                valErrs(i,split) = valErr;
                tstErrs(i,split) = tstErr;            
            end
        end

        [~, bestModelIdx] = min( mean( valErrs, 2 ) ); 
        bestC     = setOfC( bestModelIdx );
        tstErr    = mean( tstErrs( bestModelIdx, :));
        tstErrStd = std( tstErrs( bestModelIdx, :));

        save( resultFile, 'trnErrs','valErrs','tstErrs','setOfC','Opt','bestModelIdx','bestC','tstErr','tstErrStd');
    end

    Results      = load( resultFile, 'trnErrs','valErrs','tstErrs','setOfC','Opt','bestModelIdx','bestC','tstErr','tstErrStd');
    Results.file = resultFile;

    if Opt.verb
        fprintf('         C           trn_err         val_err         tst_err\n');
        for i = 1 : length( setOfC )   
            fprintf('%10.5f    %.4f(%.4f)  %.4f(%.4f)  %.4f(%.4f)',...
                Results.setOfC(i), ...
                mean(Results.trnErrs(i,:)), std(Results.trnErrs(i,:)), ...
                mean(Results.valErrs(i,:)), std(Results.valErrs(i,:)), ...
                mean(Results.tstErrs(i,:)), std(Results.tstErrs(i,:)));

            if setOfC(i) == Results.bestC, fprintf(' BEST MODEL'); end
            fprintf('\n');
        end
    end

end