function example_svm_optimal( dataSet )
% example_svm_optimal( dataSet )
%
% Train a selective classifier: linear SVM + optimal selective function
% derived from the data generating distribution.
%
% dataSet = 1;  ... difficult for max-score heuristic
% dataSet = 2;  ... easy for max-score heuristic

    rng(0);
    selclassif_setpath;


    %% Create datasets
    [trnX1,trnY1,trnX2,trnY2,valX2,valY2,tstX,tstY, Gmm] = create_gmm_data( dataSet );

    %%
    resultFolder = sprintf('results/data%d/optimal/', dataSet );

    %% Train linear SVM
    nDims  = size(trnX1,1);
    svmC   = 1;
    W      = msvmocas( [trnX1; ones(1,numel(trnY1))],trnY1,svmC);
    W      = reshape(W, nDims+1,numel(W)/(nDims+1));
    Svm.W  = W(1:end-1,:);
    Svm.W0 = W(end,:);


    %% Predict uncertainty on test examples and compute risk-coverage curve
    [predY,svmScore] = linclassif( tstX, Svm); 
    predLoss         = double( predY ~= tstY );


    nY   = numel( Gmm.Prior);
    post = exp( gmm_logpost( tstX, Gmm)); 

    uncertainty  = 1 - post( predY+[0:numel(predY)-1]*nY);

    [~,idx]      = sort( uncertainty );
    tstRiskCurve = cumsum( predLoss(idx))./[1:numel(predLoss) ];


    %% Show risk-coverage curve
    figure;
    h1=plot( 100*[1:numel(tstRiskCurve)]/numel(tstRiskCurve), tstRiskCurve);
    hold on;
    grid on;
    title('Risk-Coverage curve; uncertainty=optimal');
    xlabel('coverage [%]');
    ylabel('selective risk');
    legend(h1,sprintf('tst AUC=%.2f',mean(tstRiskCurve)));


    %% Plot decision boundary and uncertainty function
    if size(trnX1,1) == 2
        hf=figure;
        ppatterns(trnX1,trnY1);
        pclassifier( Svm, @linclassif, struct('fill',0) );
        hold on;
        title('Decision hyperplane+optimal uncertainty');

        gridDensity = 100;
        a       = axis;
        [fx,fy] = meshgrid(linspace(a(1),a(2),gridDensity),linspace(a(3),a(4),gridDensity));
        X = [fx(:)'; fy(:)'];
        [Y,svmScore] = linclassif( X, Svm); 

        post = exp( gmm_logpost( X, Gmm)); 
        uncertainty  = 1 - post( Y+[0:numel(Y)-1]*nY);

        h = contour(fx,fy, reshape(uncertainty,gridDensity,gridDensity), 'ShowText','on');
        print( hf, '-depsc', sprintf('results/svm_optimal_data%d.eps', dataSet));

    end


    %% Save results
    if ~exist( resultFolder), mkdir( resultFolder ); end
    save( [resultFolder 'results.mat'], 'tstRiskCurve');

end
