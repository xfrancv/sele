%%
% It generates EPS figures from paper to figs/ folder.

outFolder = 'figs/';

dataSet = {...
     'avila1',...
    'codrna1',...
    'covtype1',...
    'ijcnn1',...
    'letter1', ...
    'pendigit1',...
    'phishing1',...
    'sattelite1',...
    'sensorless1',...
    'shuttle1', ...
    }

legendLoc = {
    'SouthEast',...
    'NorthWest',...
    'SouthEast',...
    'NorthWest',...
    'NorthWest',...
    'NorthWest',...
    'NorthWest',...
    'NorthWest',...
    'NorthWest',...
    'NorthWest'};
 
%
if ~exist(outFolder ), mkdir( outFolder ); end

Exp1  = [];
Exp2 = [];

for i = 1 : numel( dataSet )
    Exp1(i).dataset = dataSet{i};
    Exp1(i).Result(1).name  = 'LR+plugin';
    Exp1(i).Result(1).fname = ['results/lr/' dataSet{i} '/results.mat'];

    Exp1(i).Result(end+1).name  = 'LR+sele(mlp)';
    Exp1(i).Result(end).fname   = ['results/lr/' dataSet{i} '/conf_hinge1_mlp_zmuv1/results.mat'];

    Exp1(i).Result(end+1).name  = 'LR+logistic(mlp)';
    Exp1(i).Result(end).fname   = ['results/lr/' dataSet{i} '/conf_logistic_mlp_zmuv1/results.mat'];

    
    Exp2(i).dataset = dataSet{i};
    Exp2(i).Result(1).name  = 'SVM+maxscore';
    Exp2(i).Result(1).fname   = ['results/msvmlin/' dataSet{i} '/results.mat'];

    Exp2(i).Result(end+1).name  = 'SVM+sele(mlp)';
    Exp2(i).Result(end).fname   = ['results/msvmlin/' dataSet{i} '/conf_hinge1_mlp_zmuv1/results.mat'];

    Exp2(i).Result(end+1).name  = 'SVM+logistic(mlp)';
    Exp2(i).Result(end).fname   = ['results/msvmlin/' dataSet{i} '/conf_logistic_mlp_zmuv1/results.mat'];
        
end



%
lineStyle = {'k','r','g','b','m'};
k = 0.3;
shadeColor = { [1 1 1]-k, [1 k k], [k 1 k], [k k 1]};
for e = 1 : numel( Exp1 )

    hf=figure('name', sprintf('%s', Exp1(e).dataset) );
    hold on;
    
    fprintf('\n[%s]\n', Exp1(e).dataset );
    fprintf('                                 tst AUC         R@90            R@100\n');
    h1 = [];
    str1 = [];
    maxR100=-inf;
    minR50 = inf;
    subplot(1,2,1);
    for i = 1 : numel( Exp1(e).Result )
        if exist(Exp1(e).Result(i).fname)
            R = load( Exp1(e).Result(i).fname, 'tstRiskCurve', 'tstAuc' );

            nTst = size( R.tstRiskCurve,1);
            %ciplot( min( R.tstRiskCurve, [],2), max( R.tstRiskCurve, [],2), [1:nTst]/nTst,shadeColor{i} );
            h1(end+1) = plot( [1:nTst]/nTst, mean( R.tstRiskCurve, 2), lineStyle{i}, 'linewidth', 2);
            hold on;
            str1{end+1} = Exp1(e).Result(i).name ;

            th = round( 0.9*nTst);
            fprintf('%30s   %.4f(%.4f)  %.4f(%.4f)  %.4f(%.4f)\n', Exp1(e).Result(i).name, ...
                mean(R.tstAuc), std(R.tstAuc), ...
                mean( R.tstRiskCurve(th,:)),std( R.tstRiskCurve(th,:)),...
                mean( R.tstRiskCurve(end,:)),std( R.tstRiskCurve(end,:))); 
            
            maxR100 = max(maxR100, mean( R.tstRiskCurve(end,:)));
            th = round( size(R.tstRiskCurve,1)*0.5);
            minR50  = min(minR50, mean( R.tstRiskCurve(th,:)));
        end
    end
    ha1=gca;
        
    % SVM figure
    hold on;
    
    fprintf('\n[%s]\n', Exp2(e).dataset );
    fprintf('                                 tst AUC         R@90            R@100\n');
    h2 = [];
    str2 = [];
    subplot(1,2,2);
    for i = 1 : numel( Exp2(e).Result )
        if exist(Exp2(e).Result(i).fname)
            R = load( Exp2(e).Result(i).fname, 'tstRiskCurve', 'tstAuc' );

            nTst = size( R.tstRiskCurve,1);
            h2(end+1) = plot( [1:nTst]/nTst, mean( R.tstRiskCurve, 2), lineStyle{i}, 'linewidth', 2);
            hold on;
            str2{end+1} = Exp2(e).Result(i).name ;

            th = round( 0.9*nTst);
            fprintf('%30s   %.4f(%.4f)  %.4f(%.4f)  %.4f(%.4f)\n', Exp2(e).Result(i).name, ...
                mean(R.tstAuc), std(R.tstAuc), ...
                mean( R.tstRiskCurve(th,:)),std( R.tstRiskCurve(th,:)),...
                mean( R.tstRiskCurve(end,:)),std( R.tstRiskCurve(end,:))); 
            
            maxR100 = max(maxR100, mean( R.tstRiskCurve(end,:)));
            th = round( size(R.tstRiskCurve,1)*0.5);
            minR50  = min(minR50, mean( R.tstRiskCurve(th,:)));
        end
    end
    ha2=gca;

    axes(ha1);
    grid on;
    xlabel('cover');
    ylabel('risk');
    legend( h1, str1,'Location',legendLoc{e} );
    a=axis;
    axis([0.5 a(2) minR50 maxR100]);
    h=gca;
    h.FontSize=15;
    title( Exp1(e).dataset );

    axes(ha2);
    grid on;
    xlabel('cover');
    ylabel('risk');
    legend( h2, str2,'Location',legendLoc{e} );
    a=axis;
    axis([0.5 a(2) minR50 maxR100]);
    h=gca;
    h.FontSize=15;
    hf.Position = [520 374 993 424];
    ha1.Position=[0.1300 0.1800 0.3347 0.75];
    ha2.Position=[0.5703 0.1800 0.3347 0.750];
%    print( hf, '-depsc', sprintf('%sLR+SVM_%s.eps', outFolder, Exp1(e).dataset));
    drawnow;
    snapnow;

    
end

