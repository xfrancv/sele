%% train selective classifiers: linear SVM as the predictor + SELECTIVE FUNCTION 
for data = [1 2]
    
    % SVM + optimal selection function
    example_svm_optimal( data );

    % SVM + linear selective function learned from examples
    example_svm_sele_convex( data, 'linear' );

    % SVM + quadratic selective function learned from examples
    example_svm_sele_convex( data, 'quad' );

    % SVM + multi-layer perceptron learned from examples
    example_svm_sele_mlp( data );

    % SVM + max-score used as selective fucntion
    example_svm_maxscore( data );

end

%% risk-coverage curve for various selective classifeirs
for data = [ 1 2 ]

    MaxScore = load( sprintf('results/data%d/maxscore/results.mat', data));
    SeleLin  = load( sprintf('results/data%d/sele_linear/results.mat', data));
    SeleQuad = load( sprintf('results/data%d/sele_quad/results.mat', data));
    SeleMlp  = load( sprintf('results/data%d/sele_mlp/results.mat', data));
    Optimal  = load( sprintf('results/data%d/optimal/results.mat', data));

    hf=figure('name', sprintf('data%d', data));

    h1=plot( 100*[1:numel(Optimal.tstRiskCurve)]/numel(Optimal.tstRiskCurve), Optimal.tstRiskCurve, 'r', 'linewidth',2);
    hold on;
    h2=plot( 100*[1:numel(SeleLin.tstRiskCurve)]/numel(SeleLin.tstRiskCurve), SeleLin.tstRiskCurve, 'g', 'linewidth',2);
    h3=plot( 100*[1:numel(SeleQuad.tstRiskCurve)]/numel(SeleQuad.tstRiskCurve), SeleQuad.tstRiskCurve, 'b', 'linewidth',2);
    h4=plot( 100*[1:numel(SeleMlp.tstRiskCurve)]/numel(SeleMlp.tstRiskCurve), SeleMlp.tstRiskCurve, 'c', 'linewidth',2);
    h5=plot( 100*[1:numel(MaxScore.tstRiskCurve)]/numel(MaxScore.tstRiskCurve), MaxScore.tstRiskCurve, 'k', 'linewidth',2);
    grid on;
    title('Risk-Coverage curve');
    xlabel('coverage [%]');
    ylabel('selective risk');
    legend([h1 h2 h3 h4 h5],...
        sprintf('SVM+Optimal AUC=%.2f',mean(Optimal.tstRiskCurve)),...
        sprintf('SVM+SELE(lin) AUC=%.2f',mean(SeleLin.tstRiskCurve)),...
        sprintf('SVM+SELE(quad) AUC=%.2f',mean(SeleQuad.tstRiskCurve)),...
        sprintf('SVM+Sele(mlp) AUC=%.2f',mean(SeleMlp.tstRiskCurve)),...
        sprintf('SVM+MaxScore AUC=%.2f',mean(MaxScore.tstRiskCurve)));
    set(gca,'FontSize',15);
    print( hf, '-depsc', sprintf('results/rc_data%d.eps', data));
end