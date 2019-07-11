function example_svm_sele_convex( dataSet, featureMap )
% example_svm_sele_convex( dataSet, featureMap )
%
% Train a selective classifier: linear SVM + selection function learned 
% by convex optimizatiton.
%
% dataSet = 1;  ... difficult for max-score heuristic
% dataSet = 2;  ... easy for max-score heuristic
%
% featureMap = 'linear'  ... linear selection function
% featureMap = 'quad'    ... quadratic selection function

selclassif_setpath;

lambda   = 1; % regularization constant
nBatches = 5;   % # of batches to which the risk is decomposed


%% Create datasets
[trnX1,trnY1,trnX2,trnY2,valX2,valY2,tstX,tstY] = create_gmm_data( dataSet );

%% 
resultFolder = sprintf('results/data%d/sele_%s/', dataSet, featureMap );

switch featureMap
    case 'linear'
        feature_map = @(x) x;  
    case 'quad'
        feature_map = @(x) qmap(x); 
end


%% Train linear SVM
nDims  = size(trnX1,1);
svmC   = 1;
W      = msvmocas( [trnX1; ones(1,numel(trnY1))],trnY1,svmC);
W      = reshape(W, nDims+1,numel(W)/(nDims+1));
Svm.W  = W(1:end-1,:);
Svm.W0 = W(end,:);

%% Train selection function and predict uncertainty on training examples
predY        = linclassif( trnX2, Svm); 
predLoss     = double( predY ~= trnY2 ); % 0/1-loss; but any other loss can be used as well

Sele         = train_sele_linear( feature_map(trnX2), predY, predLoss, lambda, nBatches );

uncertainty  = predict_uncertainty( feature_map(trnX2), predY, Sele );

[~,idx]      = sort( uncertainty );
trnRiskCurve = cumsum( predLoss(idx))./[1:numel(predLoss) ];


%% Predict uncertainty on test examples and compute risk-coverage curve
predY        = linclassif( tstX, Svm); 
predLoss     = double( predY ~= tstY );
uncertainty  = predict_uncertainty( feature_map(tstX), predY, Sele );
        
[~,idx]      = sort( uncertainty );
tstRiskCurve = cumsum( predLoss(idx))./[1:numel(predLoss) ];


%% Show risk-coverage curve
figure;
h1=plot( 100*[1:numel(trnRiskCurve)]/numel(trnRiskCurve), trnRiskCurve);
hold on;
h2=plot( 100*[1:numel(tstRiskCurve)]/numel(tstRiskCurve), tstRiskCurve);
grid on;
title(sprintf('Risk-Coverage curve SVM+sele(%s)', featureMap));
xlabel('coverage [%]');
ylabel('selective risk');
legend([h1 h2],sprintf('trn AUC=%.2f',mean(trnRiskCurve)),...
    sprintf('tst AUC=%.2f',mean(tstRiskCurve)));


%% Plot decision boundary and uncertainty function
if size(trnX1,1) == 2
    hf=figure;
    ppatterns(trnX1,trnY1);
    pclassifier( Svm, @linclassif, struct('fill',0) );
    hold on;
    title(sprintf('SVM classifier + learned uncertainty (%s)', featureMap));

    gridDensity = 100;
    a       = axis;
    [fx,fy] = meshgrid(linspace(a(1),a(2),gridDensity),linspace(a(3),a(4),gridDensity));
    X = [fx(:)'; fy(:)'];
    [Y,svmScore] = linclassif( X, Svm); 
    
    uncertainty = predict_uncertainty( feature_map(X), Y, Sele );
    contour(fx,fy, reshape(uncertainty,gridDensity,gridDensity),'ShowText','on','linewidth',1);
    set(gca,'FontSize',15);

    print( hf, '-depsc', sprintf('results/svm_sele_%s_data%d.eps', featureMap, dataSet));

end

%% Save results
if ~exist( resultFolder), mkdir( resultFolder ); end
save( [resultFolder 'results.mat'], 'tstRiskCurve');
