%%
% It generates EPS figures from paper to figs/ folder.

outFolder = 'figs/';

dataSet = {{'california1', [100 500 1000 5000 6190] },...
           {'abalone1', [100 500 1000 1252] },...
           {'bank1',[100 500 1000 2000 2457]},...
           {'cpu1',[100 500 1000 2000 2454] },...
           {'msd1',[100 500 1000 5000 10000]}};

legendLoc = {
    'NorthWest',...
    'NorthWest',...
    'NorthWest',...
    'NorthWest',...
    'NorthWest',...
    'NorthWest',...
    'NorthWest',...
    'NorthWest',...
    'NorthWest',...
    'NorthWest'};
 
%
if ~exist(outFolder ), mkdir( outFolder ); end


for d = 1 : numel( dataSet )
    
    
    for n = 1:numel( dataSet{d}{2})
        hf=figure;
        hold on;
        Exp  = [];
        
        Exp{1}.dataset = dataSet{d}{1};
        Exp{1}.Result(1).name  = sprintf('sele(linear),n=%d', dataSet{d}{2}(n));
        Exp{1}.Result(1).fname = sprintf('results/svorimc/%s/conf_sele1_linear_zmuv1_trn%d/results.mat',dataSet{d}{1},dataSet{d}{2}(n));

        Exp{1}.Result(end+1).name  = sprintf('reg(linear),n=%d',dataSet{d}{2}(n)) ;
        Exp{1}.Result(end).fname   = sprintf('results/svorimc/%s/conf_regression_linear_zmuv1_trn%d/results.mat',dataSet{d}{1},dataSet{d}{2}(n));

        Exp{1}.Result(end+1).name = sprintf('sele(mlp),n=%d', dataSet{d}{2}(n));
        Exp{1}.Result(end).fname  = sprintf('results/svorimc/%s/conf_sele1_mlp_zmuv1_trn%d/results.mat',dataSet{d}{1},dataSet{d}{2}(n));
        
        Exp{1}.Result(end+1).name  = sprintf('reg(mlp),n=%d', dataSet{d}{2}(n));
        Exp{1}.Result(end).fname   = sprintf('results/svorimc/%s/conf_regression_mlp_zmuv1_trn%d/results.mat',dataSet{d}{1},dataSet{d}{2}(n));
        

        %
        lineStyle = {'r','k','g','b','m','c'};


        fprintf('\n[%s]\n', dataSet{d}{1} );

        for v = 1 : numel(Exp)

%            Exp = eval(sprintf('Exp%d', v));


            fprintf('                                 tst AUC         R@90            R@100\n');
            h1 = [];
            str1 = [];
            maxR100=-inf;
            minR50 = inf;
            subplot(1,numel(Exp),v);
            tstAuc = [];
            for i = 1 : numel( Exp{v}.Result )
                if exist( Exp{v}.Result(i).fname)
                    R = load( Exp{v}.Result(i).fname, 'tstRiskCurve', 'tstAuc' );

                    nTst = size( R.tstRiskCurve,1);
                    %ciplot( min( R.tstRiskCurve, [],2), max( R.tstRiskCurve, [],2), [1:nTst]/nTst,shadeColor{i} );
                    h1(end+1) = plot( [1:nTst]/nTst, mean( R.tstRiskCurve, 2), lineStyle{i}, 'linewidth', 2);
                    hold on;
                    str1{end+1} = Exp{v}.Result(i).name ;

                    th = round( 0.9*nTst);
                    fprintf('%30s   %.4f(%.4f)  %.4f(%.4f)  %.4f(%.4f)\n', Exp{v}.Result(i).name, ...
                        mean(R.tstAuc), std(R.tstAuc), ...
                        mean( R.tstRiskCurve(th,:)),std( R.tstRiskCurve(th,:)),...
                        mean( R.tstRiskCurve(end,:)),std( R.tstRiskCurve(end,:))); 

                    maxR100 = max(maxR100, mean( R.tstRiskCurve(end,:)));
                    th = round( size(R.tstRiskCurve,1)*0.5);
                    minR50  = min(minR50, mean( R.tstRiskCurve(th,:)));

                    tstAuc(end+1) = mean(R.tstAuc);
                else
                    tstAuc(end+1) = inf;
                end
            end    
            [~,b] = min(tstAuc);
            str1{b} = [str1{b} sprintf('(BEST,auRc=%.3f)', min(tstAuc))];
            grid on;
            xlabel('cover');
            ylabel('risk');
            legend( h1, str1,'Location', legendLoc{d});
            a=axis;
    %        axis([0.5 a(2) minR50 maxR100]);
            h=gca;
            h.FontSize=15;

        end    
        title(sprintf('%s (n=%d)', dataSet{d}{1}, dataSet{d}{2}(n)));
        hf.Position = [618 396 992 815];
        drawnow;
        snapnow;

    %     ha1.Position=[0.1300 0.1800 0.3347 0.75];
    %     ha2.Position=[0.5703 0.1800 0.3347 0.750];
    %    print( hf, '-depsc', sprintf('%sLR+SVM_%s.eps', outFolder, Exp1(e).dataset));
    end
    
end
close all;
