function example_svm_sele_mlp( dataSet )
% example_svm_sele_mlp( dataSet )
%
% Train a selective classifier: linear SVM + multi-layer perceptron based 
% selective function learned from examples. 
%
% dataSet = 1;  ... difficult for max-score heuristic
% dataSet = 2;  ... easy for max-score heuristic

    %% Selective classifier based on linear SVM.
    % MLP-based selection function learned from examples.

    rng(0);
    selclassif_setpath;

    SolverOpts.numEpochs    = 100;
    SolverOpts.batchSize    = 100;
    SolverOpts.continue     = true ;
    SolverOpts.gpus         = [] ;
    SolverOpts.solver       = @solver.adam;
    SolverOpts.learningRate = 0.001;

    NetOpts.hiddenLayers = [10 10 10];
    NetOpts.dropOutRate  = [];
    NetOpts.useBatchNorm = true;


    %% Create datasets
    [trnX1,trnY1,trnX2,trnY2,valX2,valY2,tstX,tstY] = create_gmm_data( dataSet );

    %%
    resultFolder = sprintf('results/data%d/sele_mlp/', dataSet );

    %% Train linear SVM
    nDims  = size(trnX1,1);
    svmC   = 1;
    W      = msvmocas( [trnX1; ones(1,numel(trnY1))],trnY1,svmC);
    W      = reshape(W, nDims+1,numel(W)/(nDims+1));
    Svm.W  = W(1:end-1,:);
    Svm.W0 = W(end,:);

    %% Train selection function 
    nY = max( trnY2 );
    trnPredY    = linclassif( trnX2, Svm); 
    trnPredLoss = double( trnPredY ~= trnY2 ); % 0/1-loss; but any other loss can be used as well
    valPredY    = linclassif( valX2, Svm); 
    valPredLoss = double( valPredY ~= valY2 ); 
    nTrn2       = numel( trnY2 );
    nVal2       = numel( valY2 );
    nDims       = size( trnX2, 1 );

    Zmuv = zmuv( trnX2 );  % each input is normalized to have zero-mean abd unit-variance
    % Zmuv.W  = eye(2,2);
    % Zmuv.W0 = zeros(2,1);

    ImDb.images.data  = reshape( affinemap([trnX2 valX2],Zmuv), [1 1 nDims nTrn2+nVal2] );
    ImDb.images.risk  = [trnPredLoss(:) ; valPredLoss(:)];
    ImDb.images.predY = [trnPredY(:) ; valPredY(:)];
    ImDb.trnIdx       = [1:nTrn2];
    ImDb.valIdx       = [1:nVal2]+nTrn2;

    Net  =  init_confnet1( nDims, nY, NetOpts.hiddenLayers, ...
                           'dropOutRate', NetOpts.dropOutRate,...
                           'useBatchNorm', NetOpts.useBatchNorm,...
                           'leak', 0.1);
    Net.initParams();

    SolverOpts.expDir = [resultFolder 'models/'];   
    SolverOpts.train  = ImDb.trnIdx;
    SolverOpts.val    = ImDb.valIdx;
    getBatch          = @(a,b) getBatchConfDag(SolverOpts, a, b);

    [Net,Stats]  = conf_cnn_train_dag( Net, ImDb, getBatch, [], SolverOpts ) ;                                

    % load the model with smalles validation loss
    [bestEpochObjVal,bestEpoch] = min([Stats.val(:).objective]);
    Tmp  = load( sprintf('%snet-epoch-%d.mat', [resultFolder 'models/'], bestEpoch), 'net');
    Net  = dagnn.DagNN.loadobj( Tmp.net) ;

    uncertainty  = confcnn_predict( ImDb, Net, nY, 100, getBatch );   

    [~,idx]      = sort( uncertainty( ImDb.trnIdx) );
    trnRiskCurve = cumsum( trnPredLoss(idx))./[1:numel(trnPredLoss) ];

    [~,idx]      = sort( uncertainty( ImDb.valIdx) );
    valRiskCurve = cumsum( valPredLoss(idx))./[1:numel(valPredLoss) ];


    %% Predict uncertainty on test examples and compute risk-coverage curve
    [predY,svmScore] = linclassif( tstX, Svm); 
    tstPredLoss      = double( predY ~= tstY );

    ImDb.images.data  = reshape( affinemap( tstX, Zmuv), [1 1 nDims numel(tstY)]);
    ImDb.images.predY = predY;

    getBatch          = @(a,b) getBatchConfDag(SolverOpts, a, b, 1);            
    uncertainty       = confcnn_predict(ImDb, Net, nY, 100, getBatch );   

    [~,idx]           = sort( uncertainty );
    tstRiskCurve      = cumsum( tstPredLoss(idx))./[1:numel(tstPredLoss) ];


    %% Show risk-coverage curve
    figure;
    h1=plot( 100*[1:numel(trnRiskCurve)]/numel(trnRiskCurve), trnRiskCurve, 'b');
    hold on;
    h2=plot( 100*[1:numel(valRiskCurve)]/numel(valRiskCurve), valRiskCurve, 'g');
    h3=plot( 100*[1:numel(tstRiskCurve)]/numel(tstRiskCurve), tstRiskCurve, 'r');
    grid on;
    title('Risk-Coverage curve');
    xlabel('coverage [%]');
    ylabel('selective risk');
    legend([h1 h2 h3],sprintf('trn AUC=%.2f',mean(trnRiskCurve)),...
        sprintf('val AUC=%.2f',mean(valRiskCurve)), ...
        sprintf('tst AUC=%.2f',mean(tstRiskCurve)));


    %% Plot decision boundary and uncertainty function
    if size(trnX1,1) == 2
        figure;
        ppatterns(trnX1,trnY1);
        pclassifier( Svm, @linclassif, struct('fill',0) );
        hold on;
        title('Decision hyperplane+uncertainty');

        gridDensity = 100;
        a       = axis;
        [fx,fy] = meshgrid(linspace(a(1),a(2),gridDensity),linspace(a(3),a(4),gridDensity));
        X = [fx(:)'; fy(:)'];
        Y = linclassif( X, Svm); 

        ImDb.images.data  = reshape( affinemap( X, Zmuv), [1 1 nDims numel(Y)]);
        ImDb.images.predY = Y;
        getBatch          = @(a,b) getBatchConfDag(SolverOpts, a, b, 1);            
        uncertainty       = confcnn_predict(ImDb, Net, nY, 100, getBatch );   

        h = contour(fx,fy, reshape(uncertainty,gridDensity,gridDensity), 'ShowText','on');
    end

    %% Save results
    if ~exist( resultFolder), mkdir( resultFolder ); end
    save( [resultFolder 'results.mat'], 'tstRiskCurve');
end
