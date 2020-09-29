function run_train_conf_sele_mlp( dataSet, setting, trnData )

    if nargin < 1
        dataSet = 'abalone1';
        setting = 'hinge3+zmuv'; 
    end

    riskType = str2num( setting(6) );
    csetting = setting;
    csetting(6) = [];
    
    switch csetting
        case 'hinge+zmuv'
            Data        = load( ['data/' dataSet '.mat'], 'X','Y','Split' );
            rootFolder  = ['results/svorimc/' dataSet '/'];
            
            Params = [];
            for nLayers = [0 2 5]
                for batchSize = [50 100 200]
                    Params(end+1).nLayers = nLayers;
                    Params(end).batchSize = batchSize;
                    Params(end).learningRate = 0.001;
                    Params(end).dropOut      = [];
                end
            end
            numEpochs = 300;
%            riskType    = 1;
            zmuvNorm    = 1;      
        
    end

    %%
    if nargin >= 3
        Data = take_trn2_data( Data, trnData );
        outFolder = sprintf('%sconf_sele%d_mlp_zmuv%d_trn%.f/', rootFolder, riskType, zmuvNorm, trnData );
    else
        outFolder = sprintf('%sconf_sele%d_mlp_zmuv%d/', rootFolder, riskType, zmuvNorm );
    end            

    
    %%
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
            fprintf('[split=%d, param=%s]\n', split, mlp_param_str(Params(p)) );
            

            modelFile   = sprintf('%smodel_split%d_param_%s.mat', outFolder, split, mlp_param_str(Params(p)) ); 
            modelFolder = sprintf('%smodel_split%d_param_%s/', outFolder, split, mlp_param_str(Params(p)) );
            lockFile    = sprintf('%smodel_split%d_param_%s.lock', outFolder, split, mlp_param_str(Params(p)) );

            if ~exist( modelFile ) & ~exist( lockFile )
                
                fid = fopen( lockFile, 'w+');
                fprintf( fid, '%s\n%s\n', hostname, datestr(now) );            
                fclose(fid);
                
                if isempty( trnX )
                    if zmuvNorm
                        T    = zmuv( Data.X(:,Data.Split(split).trn2) );
                        trnX = affinemap(Data.X(:,Data.Split(split).trn2),T); 
                        valX = affinemap( Data.X(:,Data.Split(split).val2),T);
                    else
                        T = [];
                        trnX = Data.X(:,Data.Split(split).trn2);
                        valX = Data.X(:,Data.Split(split).val2);
                    end
                end

                %%
                nDims = size(trnX,1);
                nY    = max( Data.Y );
                
                ImDb.images.data  = reshape( [trnX valX], [1 1 nDims nTrn+nVal] );
                ImDb.images.risk  = [trnPredLoss ; valPredLoss];
                ImDb.images.predY = [trnPredY ; valPredY];
                
                ImDb.trnIdx = [1:nTrn];
                ImDb.valIdx = [1:nVal]+nTrn;
                
                %%
                if Params(p).nLayers == 0
                    nHiddenStates = [];
                else
                    nHiddenStates = nDims*ones(1,Params(p).nLayers);
                end
                
                Net  =  init_confnet1( nDims, nY, nHiddenStates, ...
                                'dropOutRate',  Params(p).dropOut,...
                                'useBatchNorm', true,...
                                'leak', 0.1,...
                                'loss', riskType);
                Net.initParams();

                % Meta ... info about data and the prediction model
                Meta.nInputs  = nDims;
                Meta.nOutputs = nY;
                
                Opts.expDir        = modelFolder;
                Opts.numEpochs     = numEpochs;
                Opts.batchSize     = Params(p).batchSize ;
                Opts.continue      = true ;
                Opts.gpus          = [] ;
                Opts.solver        = @solver.adam;
                Opts.learningRate  = Params(p).learningRate;
                
                Opts.train = ImDb.trnIdx;
                Opts.val   = ImDb.valIdx;
                getBatch   = @(a,b) getBatchConfDag(Opts, a, b);

                [Net,Stats]  = conf_cnn_train_dag( Net, ImDb, getBatch, Meta, Opts ) ;                                
                
                % load the best model
                [bestEpochAuRC,bestEpoch] = min([Stats.val(:).auRC]);
                Tmp  = load( sprintf('%snet-epoch-%d.mat', modelFolder, bestEpoch), 'net');
                Net  = dagnn.DagNN.loadobj( Tmp.net) ;
                
                conf = confcnn_predict( ImDb, Net, nY, 100, getBatch );   


                [~,idx] = sort( conf( ImDb.valIdx) );
                valRiskCurve = cumsum( valPredLoss(idx))./[1:nVal]';
                valAuc  = mean( valRiskCurve );                
                valLoss  = sum( cumsum( valPredLoss(idx) ))/(nVal^2);
                
                [~,idx] = sort( conf(ImDb.trnIdx) );
                trnRiskCurve = cumsum( trnPredLoss(idx))./[1:nTrn]';
                trnAuc  = mean( trnRiskCurve );                
                trnLoss  = sum( cumsum( trnPredLoss(idx)))/(nTrn^2);
                

                fprintf('trnLoss=%.4f, valLoss =%.4f\n', trnLoss, valLoss);

                save( modelFile, 'Net', 'trnLoss', 'valLoss', 'T', 'trnAuc', 'valAuc' );
                
                delete( lockFile );
            end
        end    
    end

    
    %%
    numDone    = 0;
    numMissing = 0;
    for split = 1 : nSplits
        for p = 1 : numel( Params) 
            modelFile   = sprintf('%smodel_split%d_param_%s.mat', outFolder, split, mlp_param_str(Params(p)) ); 
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

    %
    % erase intermediate NN models
    for split = 1 : nSplits
        for p = 1 : numel( Params) 
            modelFolder = sprintf('%smodel_split%d_param_%s/', outFolder, split, mlp_param_str(Params(p)) );
            modelFile   = sprintf('%smodel_split%d_param_%s.mat', outFolder, split, mlp_param_str(Params(p)) ); 
            if exist( modelFile )
                list = dir( [modelFolder '*.mat']);
                fprintf('Erasing %d files from %s\n', numel(list), modelFolder );
                for i = 1 : numel( list )
                    delete([list(i).folder '/' list(i).name]);
                end
            end
        end
    end
    
    %% Collect results
    trnAuc = zeros( numel( Params ), nSplits );
    valAuc = zeros( numel( Params ), nSplits );
    for p = 1 : numel(  Params )
        for split = 1 : nSplits
            modelFile   = sprintf('%smodel_split%d_param_%s.mat', outFolder, split, mlp_param_str(Params(p)) );
            R = load( modelFile, 'trnAuc', 'valAuc' );

            trnAuc( p, split) = R.trnAuc;
            valAuc( p, split) = R.valAuc;
        end
    end

    %% Find best lambda
    fprintf('split   param                          trnAuc   valAuc\n');
    bestParams   = nan*ones(nSplits,1);
    for split = 1 : nSplits
        [~,idx ] = min( valAuc(:,split));
        bestParams(split) = idx;
        fprintf('%d        %2d   %30s           %.4f   %.4f\n', split, idx, ...
            mlp_param_str(Params(idx)), trnAuc(idx,split), valAuc(idx,split));
    end

    fprintf('    param    trnAuc          valAuc\n');
    for p = 1 : numel( Params )
        fprintf('%2d   %30s  %.4f(%.4f)  %.4f(%.4f)\n', p, mlp_param_str(Params(p)), ...
            mean( trnAuc(p,:)), std( trnAuc(p,:)), ...
            mean( valAuc(p,:)), std( valAuc(p,:)) );
    end

    %% Evaluate best model on test data
    for split = 1 : nSplits

        predFile   = sprintf('%sresults_split%d.mat', rootFolder, split);         
        modelFile  = sprintf('%smodel_split%d_param_%s.mat', outFolder, split, mlp_param_str(Params(bestParams(split))) );
        resultFile = sprintf('%sresults_split%d.mat', outFolder, split );

        Pred     = load( predFile );
        predLoss = Pred.predLoss;
        predY    = Pred.predY;

        tstIdx = Data.Split(split).tst;
        nTst   = numel( tstIdx );
        valIdx = Data.Split(split).val2;
        nVal   = numel( valIdx );
        nDims  = size( Data.X,1);


        if ~exist( resultFile )       

            Tmp = load( modelFile, 'Net', 'T' );
            Net  = dagnn.DagNN.loadobj( Tmp.Net) ;
            T    = Tmp.T;

            if zmuvNorm
                ImDb.images.data = reshape(affinemap( Data.X, T), [1 1 nDims size(Data.X,2)]);
                ImDb.images.predY = predY;
            else
                Imdb.images.data  = reshape(Data.X, [1 1 nDims size(Data.X,2)]);
                ImDb.images.predY = predY;
            end
    
            Opts.gpus      = [];
            getBatch       = @(a,b) getBatchConfDag(Opts, a, b, 1);            
            uncertainty    = confcnn_predict(ImDb, Net, nY, 100, getBatch );   
            
            tstPredLoss    = predLoss( tstIdx);
            [~,idx]        = sort( uncertainty( tstIdx) );
            tstRiskCurve   = cumsum( tstPredLoss(idx))./[1:nTst]';
            tstAuc         = mean( tstRiskCurve);
            tstLoss        = sum( cumsum( tstPredLoss(idx) ))/(nTst^2);

            valPredLoss    = predLoss( valIdx);
            [~,idx]        = sort( uncertainty( valIdx) );
            valLoss        = sum( cumsum( valPredLoss(idx) ))/(nVal^2);
            
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
        R          = load( resultFile, 'tstAuc', 'tstRiskCurve','valLoss', 'tstLoss' );

        tstAuc       = [tstAuc R.tstAuc ];
        tstLoss      = [tstLoss R.tstLoss];
        valLoss      = [valLoss R.valLoss ];
        tstRiskCurve = [tstRiskCurve R.tstRiskCurve];
    end

    fprintf('tstAuc=%.4f(%.4f), valLoss=%.4f(%.4f)\n', mean( tstAuc), std( tstAuc ), mean( valLoss), std( valLoss ));

    outFile = [outFolder 'results.mat'];
    save( outFile, 'tstRiskCurve', 'tstAuc','tstLoss', 'valLoss' );

    %%
% 
    figure;
    plot(  [1:nTst]/nTst, tstRiskCurve );
    hold on;
    plot( [1:nTst]/nTst, mean( tstRiskCurve, 2), 'r', 'linewidth', 2);
    xlabel('cover');
    ylabel('err');
    grid on;
    
    %
    return;
end


