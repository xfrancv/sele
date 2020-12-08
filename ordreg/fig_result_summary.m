%%

showLinear = 1;
showQuad = 1;
showMlp = 0;
showSele2 = 1;

dataSet = {{'california1', [100 500 1000 5000 6190] },...
           {'abalone1', [100 500 1000 1252] },...
           {'bank1',[100 500 1000 2000 2457]},...
           {'cpu1',[100 500 1000 2000 2454] },...
           {'bikeshare1', [100 500 1000 5213] },...
           {'ccpp1', [100 500 1000 2872] },...
           {'facebook1',[100 500 1000 5000 10000]},...
           {'gpu1',[100 500 1000 5000 10000] },...
           {'superconduct1',[100 500 1000 5000 6378]},...
           {'metro1',[100 500 1000 5000 10000] }};
%           {'msd1',[100 500 1000 5000 10000]},...
%           {'metro1',[100 500 1000 5000 10000] }};
           

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
 

COVER = 0.9;
Result = [];
for d = 1 : numel( dataSet )
    

    n = numel( dataSet{d}{2});
    nTrn = dataSet{d}{2}(n);
    dataset = dataSet{d}{1};

    Result{d}  = [];
    
    if showLinear
        Result{d}(end+1).name  = sprintf('sele(linear)');
        Result{d}(end).fname = sprintf('results/svorimc/%s/conf_sele1_linear_zmuv1_trn%d/results.mat',dataSet{d}{1},dataSet{d}{2}(n));

        if showSele2
            Result{d}(end+1).name  = sprintf('sele2(linear)');
            Result{d}(end).fname = sprintf('results/svorimc/%s/conf_sele2_linear_zmuv1_trn%d/results.mat',dataSet{d}{1},dataSet{d}{2}(n));
        end
        
        Result{d}(end+1).name  = sprintf('reg(linear)') ;
        Result{d}(end).fname   = sprintf('results/svorimc/%s/conf_regression_linear_zmuv1_trn%d/results.mat',dataSet{d}{1},dataSet{d}{2}(n));
        
    end
    
    if showMlp
        Result{d}(end+1).name = sprintf('sele(mlp)');
        Result{d}(end).fname  = sprintf('results/svorimc/%s/conf_sele1_mlp_zmuv1_trn%d/results.mat',dataSet{d}{1},dataSet{d}{2}(n));

        Result{d}(end+1).name  = sprintf('reg(mlp)');
        Result{d}(end).fname   = sprintf('results/svorimc/%s/conf_regression_mlp_zmuv1_trn%d/results.mat',dataSet{d}{1},dataSet{d}{2}(n));
    end

    if showQuad
        Result{d}(end+1).name = sprintf('sele(quad)');
        Result{d}(end).fname  = sprintf('results/svorimc/%s/conf_sele1_quad_zmuv1_trn%d/results.mat',dataSet{d}{1},dataSet{d}{2}(n));
        
        if showSele2
            Result{d}(end+1).name = sprintf('sele2(quad)');
            Result{d}(end).fname  = sprintf('results/svorimc/%s/conf_sele2_quad_zmuv1_trn%d/results.mat',dataSet{d}{1},dataSet{d}{2}(n));
        end

        Result{d}(end+1).name = sprintf('reg(quad)');
        Result{d}(end).fname  = sprintf('results/svorimc/%s/conf_regression_quad_zmuv1_trn%d/results.mat',dataSet{d}{1},dataSet{d}{2}(n));
    end
end


%% Table 
for d = 1 : numel( dataSet )

    n = numel( dataSet{d}{2});
    nTrn = dataSet{d}{2}(n);
    dataset = dataSet{d}{1};
    
    if d == 1
        fprintf('               ');
        for i = 1 : numel( Result{d}) 
            fprintf('| %28s   ', Result{d}(i).name );
        end
        fprintf('|\n  ');
        fprintf('             ');
        for i = 1 : numel( Result{d} ) 
            fprintf('| AuRC            R@%2d           ', round(100*COVER) );
        end
        fprintf('| R@100\n');
        
        
    end
            
    fprintf('%14s ', dataset);
    R = [];
    aurc = [];
    rat = [];
    for i = 1 : numel( Result{d}) 
        if exist( Result{d}(i).fname)
            R{i} = load( Result{d}(i).fname, 'tstRiskCurve', 'tstAuc' );
            nTst = size( R{i}.tstRiskCurve,1);        
            th = round( COVER*nTst);
            aurc(i) = mean(R{i}.tstAuc);
            rat(i)  = mean( R{i}.tstRiskCurve(th,:));
        else
            aurc(i) = inf;
            rat(i)=inf;
        end
    end
        
    for i = 1 : numel( Result{d} ) 
        
        l1=' ';
        r1= ' ';
        l2=' ';
        r2=' ';
        if aurc(i) < inf
            nTst = size( R{i}.tstRiskCurve,1);        
            th = round( COVER*nTst);

            if mean(R{i}.tstAuc) == min( aurc )
                l1='[';
                r1=']';
            end
            if mean( R{i}.tstRiskCurve(th,:)) == min( rat)
                l2='[';
                r2=']';
            end

            fprintf('|%c%.4f(%.4f)%c%c%.4f(%.4f)%c',...
                    l1,mean(R{i}.tstAuc), std(R{i}.tstAuc), r1,l2,...
                    mean( R{i}.tstRiskCurve(th,:)),std( R{i}.tstRiskCurve(th,:)),r2 );
        else
            fprintf('|%c%.4f(%.4f)%c%c%.4f(%.4f)%c',...
                    l1,nan, nan, r1,l2,...
                    nan,nan,r2 );
        end
            
    end
    fprintf('| %.4f(%.4f)\n', mean( R{1}.tstRiskCurve(end,:)),std( R{1}.tstRiskCurve(end,:)));
    
end
%        

% plot all figures
for d = 1 : numel( dataSet )
    
    n = numel( dataSet{d}{2});
    nTrn = dataSet{d}{2}(n);
    dataset = dataSet{d}{1};
    
    for n = numel( dataSet{d}{2})
        
        hf=figure;
        hold on;
        
        dataset = dataSet{d}{1};

        %
        lineStyle = {'r','k','g','b','m','c'};

        fprintf('\n[%s]\n', dataSet{d}{1} );

        fprintf('                                 tst AUC         R@90            R@100\n');
        h1 = [];
        str1 = [];
        maxR100=-inf;
        minR50 = inf;
        tstAuc = [];
        for i = 1 : numel( Result{d} )
            if exist( Result{d}(i).fname)
                R = load( Result{d}(i).fname, 'tstRiskCurve', 'tstAuc' );

                nTst = size( R.tstRiskCurve,1);
                h1(end+1) = plot( [1:nTst]/nTst, mean( R.tstRiskCurve, 2), lineStyle{i}, 'linewidth', 2);
                hold on;
                str1{end+1} = Result{d}(i).name ;

                th = round( COVER*nTst);
                fprintf('%30s   %.4f(%.4f)  %.4f(%.4f)  %.4f(%.4f)\n', Result{d}(i).name, ...
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

        title(sprintf('%s (n=%d)', dataSet{d}{1}, dataSet{d}{2}(n)));
        hf.Position = [618 396 992 815];
        drawnow;
        snapnow;

        
    end
    
end
close all;
