

for data = {'data1','data2'}

MaxScore = load( sprintf('results/%s/maxscore/results.mat', data{1}));
SeleLin  = load( sprintf('results/%s/sele_linear/results.mat', data{1}));
SeleQuad = load( sprintf('results/%s/sele_quad/results.mat', data{1}));
SeleMlp  = load( sprintf('results/%s/sele_mlp/results.mat', data{1}));
Optimal  = load( sprintf('results/%s/optimal/results.mat', data{1}));

    hf=figure('name', data{1});

    h1=plot( 100*[1:numel(MaxScore.tstRiskCurve)]/numel(MaxScore.tstRiskCurve), MaxScore.tstRiskCurve, 'k', 'linewidth',2);
    hold on;
    h2=plot( 100*[1:numel(SeleLin.tstRiskCurve)]/numel(SeleLin.tstRiskCurve), SeleLin.tstRiskCurve, 'g', 'linewidth',2);
    h3=plot( 100*[1:numel(SeleQuad.tstRiskCurve)]/numel(SeleQuad.tstRiskCurve), SeleQuad.tstRiskCurve, 'b', 'linewidth',2);
    h4=plot( 100*[1:numel(SeleQuad.tstRiskCurve)]/numel(SeleMlp.tstRiskCurve), SeleMlp.tstRiskCurve, 'm', 'linewidth',2);
    h5=plot( 100*[1:numel(Optimal.tstRiskCurve)]/numel(Optimal.tstRiskCurve), Optimal.tstRiskCurve, 'r', 'linewidth',2);
    grid on;
    title('Risk-Coverage curve');
    xlabel('coverage [%]');
    ylabel('selective risk');
    legend([h1 h2 h3 h4 h5],sprintf('SVM+MaxScore AUC=%.2f',mean(MaxScore.tstRiskCurve)),...
        sprintf('SVM+Sele(lin) AUC=%.2f',mean(SeleLin.tstRiskCurve)),...
        sprintf('SVM+Sele(quad) AUC=%.2f',mean(SeleQuad.tstRiskCurve)),...
        sprintf('SVM+Sele(mlp) AUC=%.2f',mean(SeleMlp.tstRiskCurve)),...
        sprintf('SVM+Optimal AUC=%.2f',mean(Optimal.tstRiskCurve)));

    print( hf, '-depsc', sprintf('results/rc_%s.eps', data{1}));
end