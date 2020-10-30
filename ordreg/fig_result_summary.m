%%

dataSet = {{'california1', [100 500 1000 5000 6190] },...
           {'abalone1', [100 500 1000 1252] },...
           {'bank1',[100 500 1000 2000 2457]},...
           {'cpu1',[100 500 1000 2000 2454] },...
           {'msd1',[100 500 1000 5000 10000]},...
           {'bikeshare1', [100 500 1000 5213] },...
           {'ccpp1', [100 500 1000 2872] },...
           {'facebook1',[100 500 1000 5000 10000]},...
           {'gpu1',[100 500 1000 5000 10000] },...
           {'superconduct1',[100 500 1000 5000 6378]}};
           

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
 

COVER = 0.5;
% draw table
for d = 1 : numel( dataSet )
    
    Result  = [];

    n = numel( dataSet{d}{2});
    nTrn = dataSet{d}{2}(n);
    dataset = dataSet{d}{1};
    
    Result(1).name  = sprintf('sele(linear)');
    Result(1).fname = sprintf('results/svorimc/%s/conf_sele1_linear_zmuv1_trn%d/results.mat',dataSet{d}{1},dataSet{d}{2}(n));

    Result(end+1).name  = sprintf('reg(linear)') ;
    Result(end).fname   = sprintf('results/svorimc/%s/conf_regression_linear_zmuv1_trn%d/results.mat',dataSet{d}{1},dataSet{d}{2}(n));

    Result(end+1).name = sprintf('sele(mlp)');
    Result(end).fname  = sprintf('results/svorimc/%s/conf_sele1_mlp_zmuv1_trn%d/results.mat',dataSet{d}{1},dataSet{d}{2}(n));

    Result(end+1).name  = sprintf('reg(mlp)');
    Result(end).fname   = sprintf('results/svorimc/%s/conf_regression_mlp_zmuv1_trn%d/results.mat',dataSet{d}{1},dataSet{d}{2}(n));

%     Result(end+1).name = sprintf('sele(quad)');
%     Result(end).fname  = sprintf('results/svorimc/%s/conf_sele1_quad_zmuv1_trn%d/results.mat',dataSet{d}{1},dataSet{d}{2}(n));
% 
%     Result(end+1).name = sprintf('reg(quad)');
%     Result(end).fname  = sprintf('results/svorimc/%s/conf_regression1_quad_zmuv1_trn%d/results.mat',dataSet{d}{1},dataSet{d}{2}(n));

    if d == 1
        fprintf('               ');
        for i = 1 : numel( Result) 
            fprintf('| %28s   ', Result(i).name );
        end
        fprintf('|\n  ');
        fprintf('             ');
        for i = 1 : numel( Result) 
            fprintf('| AuRC            R@%2d           ', round(100*COVER) );
        end
        fprintf('| R@100\n');
        
        
    end
            
    fprintf('%14s ', dataset);
    R = [];
    aurc = [];
    rat = [];
    for i = 1 : numel( Result) 
        if exist( Result(i).fname)
            R{i} = load( Result(i).fname, 'tstRiskCurve', 'tstAuc' );
            nTst = size( R{i}.tstRiskCurve,1);        
            th = round( COVER*nTst);
            aurc(i) = mean(R{i}.tstAuc);
            rat(i)  = mean( R{i}.tstRiskCurve(th,:));
        else
            aurc(i) = inf;
            rat(i)=inf;
        end
    end
        
    for i = 1 : numel( Result) 
        
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
    
    for n = numel( dataSet{d}{2})
        
        hf=figure;
        hold on;
        Result  = [];
        
        dataset = dataSet{d}{1};
        Result(1).name  = sprintf('sele(linear),n=%d', dataSet{d}{2}(n));
        Result(1).fname = sprintf('results/svorimc/%s/conf_sele1_linear_zmuv1_trn%d/results.mat',dataSet{d}{1},dataSet{d}{2}(n));

        Result(end+1).name  = sprintf('reg(linear),n=%d',dataSet{d}{2}(n)) ;
        Result(end).fname   = sprintf('results/svorimc/%s/conf_regression_linear_zmuv1_trn%d/results.mat',dataSet{d}{1},dataSet{d}{2}(n));

        Result(end+1).name = sprintf('sele(mlp),n=%d', dataSet{d}{2}(n));
        Result(end).fname  = sprintf('results/svorimc/%s/conf_sele1_mlp_zmuv1_trn%d/results.mat',dataSet{d}{1},dataSet{d}{2}(n));
        
        Result(end+1).name  = sprintf('reg(mlp),n=%d', dataSet{d}{2}(n));
        Result(end).fname   = sprintf('results/svorimc/%s/conf_regression_mlp_zmuv1_trn%d/results.mat',dataSet{d}{1},dataSet{d}{2}(n));

%         Result(end+1).name = sprintf('sele(quad)');
%         Result(end).fname  = sprintf('results/svorimc/%s/conf_sele1_quad_zmuv1_trn%d/results.mat',dataSet{d}{1},dataSet{d}{2}(n));
% 
%         Result(end+1).name = sprintf('reg(quad)');
%         Result(end).fname  = sprintf('results/svorimc/%s/conf_regression1_quad_zmuv1_trn%d/results.mat',dataSet{d}{1},dataSet{d}{2}(n));
        

        %
        lineStyle = {'r','k','g','b','m','c'};


        fprintf('\n[%s]\n', dataSet{d}{1} );


        fprintf('                                 tst AUC         R@90            R@100\n');
        h1 = [];
        str1 = [];
        maxR100=-inf;
        minR50 = inf;
        tstAuc = [];
        for i = 1 : numel( Result )
            if exist( Result(i).fname)
                R = load( Result(i).fname, 'tstRiskCurve', 'tstAuc' );

                nTst = size( R.tstRiskCurve,1);
                h1(end+1) = plot( [1:nTst]/nTst, mean( R.tstRiskCurve, 2), lineStyle{i}, 'linewidth', 2);
                hold on;
                str1{end+1} = Result(i).name ;

                th = round( COVER*nTst);
                fprintf('%30s   %.4f(%.4f)  %.4f(%.4f)  %.4f(%.4f)\n', Result(i).name, ...
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
