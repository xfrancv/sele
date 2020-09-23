%%
% It generates EPS figures from paper to figs/ folder.

outFolder = 'figs/';

dataSet = {{'california1', [100 500 1000 5000 6190] },...
           {'abalone1', [100 500 1000 1252] },...
           {'bank1',[100 500 1000 2000 2457]},...
           {'cpu1',[100 500 1000 2000 2454] }};


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
 
%%
if ~exist(outFolder ), mkdir( outFolder ); end

Exp1  = [];
Exp2 = [];
Exp3 = [];

for i = 1 : numel( dataSet )
    Exp1(i).dataset = dataSet{i}{1};
    Exp1(i).Result(1).name  = 'SVORIMC+sele(linear)-100';
    Exp1(i).Result(1).fname = sprintf('results/svorimc/%s/conf_sele1_linear_zmuv1_trn%d/results.mat',dataSet{i}{1},dataSet{i}{2}(1));
    Exp1(i).Result(end+1).name  = 'SVORIMC+sele(linear)-all';
    Exp1(i).Result(end).fname = sprintf('results/svorimc/%s/conf_sele1_linear_zmuv1_trn%d/results.mat',dataSet{i}{1},dataSet{i}{2}(end));

    Exp1(i).Result(end+1).name  = 'SVORIMC+regression(linear)-100';
    Exp1(i).Result(end).fname   = sprintf('results/svorimc/%s/conf_regression_linear_zmuv1_trn%d/results.mat',dataSet{i}{1},dataSet{i}{2}(1));
    Exp1(i).Result(end+1).name  = 'SVORIMC+regression(linear)-all';
    Exp1(i).Result(end).fname   = sprintf('results/svorimc/%s/conf_regression_linear_zmuv1_trn%d/results.mat',dataSet{i}{1},dataSet{i}{2}(end));

    Exp2(i).dataset = dataSet{i}{1};
    Exp2(i).Result(1).name = 'SVORIMC+sele(quad)-100';
    Exp2(i).Result(1).fname  = sprintf('results/svorimc/%s/conf_sele1_quad_zmuv1_trn%d/results.mat',dataSet{i}{1},dataSet{i}{2}(1));
    Exp2(i).Result(end+1).name = 'SVORIMC+sele(quad)-all';
    Exp2(i).Result(end).fname  = sprintf('results/svorimc/%s/conf_sele1_quad_zmuv1_trn%d/results.mat',dataSet{i}{1},dataSet{i}{2}(end));

    Exp2(i).Result(end+1).name  = 'SVORIMC+regression(quad)-100';
    Exp2(i).Result(end).fname   = sprintf('results/svorimc/%s/conf_regression_quad_zmuv1_trn%d/results.mat',dataSet{i}{1},dataSet{i}{2}(1));
    Exp2(i).Result(end+1).name  = 'SVORIMC+regression(quad)-all';
    Exp2(i).Result(end).fname   = sprintf('results/svorimc/%s/conf_regression_quad_zmuv1_trn%d/results.mat',dataSet{i}{1},dataSet{i}{2}(end));

    
    Exp3(i).dataset = dataSet{i}{1};
    Exp3(i).Result(1).name = 'SVORIMC+sele(mlp)-100';
    Exp3(i).Result(1).fname  = sprintf('results/svorimc/%s/conf_sele1_mlp_zmuv1_trn%d/results.mat',dataSet{i}{1},dataSet{i}{2}(1));
    Exp3(i).Result(end+1).name = 'SVORIMC+sele(mlp)-all';
    Exp3(i).Result(end).fname  = sprintf('results/svorimc/%s/conf_sele1_mlp_zmuv1_trn%d/results.mat',dataSet{i}{1},dataSet{i}{2}(end));

    Exp3(i).Result(end+1).name  = 'SVORIMC+regression(mlp)-100';
    Exp3(i).Result(end).fname   = sprintf('results/svorimc/%s/conf_regression_mlp_zmuv1_trn%d/results.mat',dataSet{i}{1},dataSet{i}{2}(1));
    Exp3(i).Result(end+1).name  = 'SVORIMC+regression(mlp)-all';
    Exp3(i).Result(end).fname   = sprintf('results/svorimc/%s/conf_regression_mlp_zmuv1_trn%d/results.mat',dataSet{i}{1},dataSet{i}{2}(end));

    
end



%%
lineStyle = {'k','r','g','b','m','c'};
k = 0.3;
shadeColor = { [1 1 1]-k, [1 k k], [k 1 k], [k k 1]};
for e = 1 : numel( Exp1 )

    hf=figure;
 %   title( sprintf('LR on %s', Exp1(e).dataset) );
    hold on;
    
    fprintf('\n[%s]\n', Exp1(e).dataset );
    
    for v = 1 : 3
    
        Exp = eval(sprintf('Exp%d', v));
        
    
        fprintf('                                 tst AUC         R@90            R@100\n');
        h1 = [];
        str1 = [];
        maxR100=-inf;
        minR50 = inf;
        subplot(1,3,v);
        for i = 1 : numel( Exp(e).Result )
            if exist( Exp(e).Result(i).fname)
                R = load( Exp(e).Result(i).fname, 'tstRiskCurve', 'tstAuc' );

                nTst = size( R.tstRiskCurve,1);
                %ciplot( min( R.tstRiskCurve, [],2), max( R.tstRiskCurve, [],2), [1:nTst]/nTst,shadeColor{i} );
                h1(end+1) = plot( [1:nTst]/nTst, mean( R.tstRiskCurve, 2), lineStyle{i}, 'linewidth', 2);
                hold on;
                str1{end+1} = Exp(e).Result(i).name ;

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
        grid on;
        xlabel('cover');
        ylabel('risk');
        legend( h1, str1,'Location',legendLoc{e} );
        a=axis;
%        axis([0.5 a(2) minR50 maxR100]);
        h=gca;
        h.FontSize=15;

    end    
    
    
    hf.Position = [520 274 1673 524];
%     ha1.Position=[0.1300 0.1800 0.3347 0.75];
%     ha2.Position=[0.5703 0.1800 0.3347 0.750];
%    print( hf, '-depsc', sprintf('%sLR+SVM_%s.eps', outFolder, Exp1(e).dataset));
    
end

