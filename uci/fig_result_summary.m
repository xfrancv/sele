%%

showLinear = 1;
showQuad = 0;
showMlp = 0;
showSele2 = 1;
showSele3 = 1;

outFolder = 'figs/';

dataSet = {{'avila1', [100 500 1000 5000]}, ...
           {'codrna1',[100 500 1000 5000 10000 15000]}, ...
           {'covtype1',[100 500 1000 5000 10000]}, ...
           {'ijcnn1',[100 500 1000 5000 10000 14000]},...
           {'letter1',[100 500 1000 5000]},...
           {'pendigit1',[100 500 1000 3000]},...
           {'phishing1',[100 500 1000 3000]},...
           {'sattelite1',[100 500 1000 1500]},...
           {'sensorless1',[100 500 1000 5000 10000 15000]},...
           {'shuttle1',[100 500 1000 5000 10000 15000]} };

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


for i = 1 : numel( dataSet )
    Exp1(i).dataset = dataSet{i}{1};
    Exp1(i).Result(1).name  = 'LR+plugin';
    Exp1(i).Result(1).fname = ['results/lr/' dataSet{i}{1} '/results.mat'];

    if showLinear 
        Exp1(i).Result(end+1).name = 'LR+sele1(linear)';
        Exp1(i).Result(end).fname  = ['results/lr/' dataSet{i}{1} sprintf('/conf_sele1_linear_zmuv1_trn%d/results.mat',dataSet{i}{2}(end))];

        if showSele2
            Exp1(i).Result(end+1).name = 'LR+sele2(linear)';
            Exp1(i).Result(end).fname  = ['results/lr/' dataSet{i}{1} sprintf('/conf_sele2_linear_zmuv1_trn%d/results.mat',dataSet{i}{2}(end))];
        end
        if showSele3
            Exp1(i).Result(end+1).name = 'LR+sele3(linear)';
            Exp1(i).Result(end).fname  = ['results/lr/' dataSet{i}{1} sprintf('/conf_sele3_linear_zmuv1_trn%d/results.mat',dataSet{i}{2}(end))];
        end

        Exp1(i).Result(end+1).name = 'LR+logistic(linear)';
        Exp1(i).Result(end).fname  = ['results/lr/' dataSet{i}{1} sprintf('/conf_logistic_linear_zmuv1_trn%d/results.mat',dataSet{i}{2}(end))];
    end

    if showQuad
        Exp1(i).Result(end+1).name  = 'LR+sele1(quad)';
        Exp1(i).Result(end).fname   = ['results/lr/' dataSet{i}{1} sprintf('/conf_sele1_quad_zmuv1_trn%d/results.mat',dataSet{i}{2}(end))];
        Exp1(i).Result(end+1).name = 'LR+logistic(quad)';
        Exp1(i).Result(end).fname  = ['results/lr/' dataSet{i}{1} sprintf('/conf_logistic_quad_zmuv1_trn%d/results.mat',dataSet{i}{2}(end))];
    end
        
    

    if showMlp
        Exp1(i).Result(end+1).name  = 'LR+sele1(mlp)';
        Exp1(i).Result(end).fname   = ['results/lr/' dataSet{i}{1} sprintf('/conf_sele1_mlp_zmuv1_trn%d/results.mat',dataSet{i}{2}(end))];
        if showSele2
            Exp1(i).Result(end+1).name  = 'LR+sele2(mlp)';
            Exp1(i).Result(end).fname   = ['results/lr/' dataSet{i}{1} sprintf('/conf_sele2_mlp_zmuv1_trn%d/results.mat',dataSet{i}{2}(end))];
        end
        Exp1(i).Result(end+1).name = 'LR+logistic(mlp)';
        Exp1(i).Result(end).fname  = ['results/lr/' dataSet{i}{1} sprintf('/conf_logistic_mlp_zmuv1_trn%d/results.mat',dataSet{i}{2}(end))];
    end

    
    Exp2(i).dataset = dataSet{i}{1};
    Exp2(i).Result(1).name  = 'SVM+maxscore';
    Exp2(i).Result(1).fname   = ['results/msvmlin/' dataSet{i}{1} '/results.mat'];

    if showLinear
        Exp2(i).Result(end+1).name  = 'SVM+sele1(linear)';
        Exp2(i).Result(end).fname   = ['results/msvmlin/' dataSet{i}{1} sprintf('/conf_sele1_linear_zmuv1_trn%d/results.mat',dataSet{i}{2}(end))];
        if showSele2
            Exp2(i).Result(end+1).name  = 'SVM+sele2(linear)';
            Exp2(i).Result(end).fname   = ['results/msvmlin/' dataSet{i}{1} sprintf('/conf_sele2_linear_zmuv1_trn%d/results.mat',dataSet{i}{2}(end))];
        end
        if showSele3
            Exp2(i).Result(end+1).name  = 'SVM+sele3(linear)';
            Exp2(i).Result(end).fname   = ['results/msvmlin/' dataSet{i}{1} sprintf('/conf_sele3_linear_zmuv1_trn%d/results.mat',dataSet{i}{2}(end))];
        end
        Exp2(i).Result(end+1).name  = 'SVM+logistic1(linear)';
        Exp2(i).Result(end).fname   = ['results/msvmlin/' dataSet{i}{1} sprintf('/conf_logistic_linear_zmuv1_trn%d/results.mat',dataSet{i}{2}(end))];
    end

     if showQuad
         Exp2(i).Result(end+1).name  = 'SVM+sele1(quad)';
         Exp2(i).Result(end).fname   = ['results/msvmlin/' dataSet{i}{1} sprintf('/conf_sele1_quad_zmuv1_trn%d/results.mat',dataSet{i}{2}(end))];
         Exp2(i).Result(end+1).name  = 'SVM+logistic1(quad)';
         Exp2(i).Result(end).fname   = ['results/msvmlin/' dataSet{i}{1} sprintf('/conf_logistic_quad_zmuv1_trn%d/results.mat',dataSet{i}{2}(end))];
     end
     
     if showMlp
         Exp2(i).Result(end+1).name  = 'SVM+sele1(mlp)';
         Exp2(i).Result(end).fname   = ['results/msvmlin/' dataSet{i}{1} sprintf('/conf_sele1_mlp_zmuv1_trn%d/results.mat',dataSet{i}{2}(end))];

         if showSele2
             Exp2(i).Result(end+1).name  = 'SVM+sele2(mlp)';
             Exp2(i).Result(end).fname   = ['results/msvmlin/' dataSet{i}{1} sprintf('/conf_sele2_mlp_zmuv1_trn%d/results.mat',dataSet{i}{2}(end))];
         end

         Exp2(i).Result(end+1).name  = 'SVM+logistic1(mlp)';
         Exp2(i).Result(end).fname   = ['results/msvmlin/' dataSet{i}{1} sprintf('/conf_logistic_mlp_zmuv1_trn%d/results.mat',dataSet{i}{2}(end))];
     end
end



%%
lineStyle = {'k','r','g','b','m','y'};
k = 0.3;
shadeColor = { [1 1 1]-k, [1 k k], [k 1 k], [k k 1]};
for e = 1 : numel( Exp1 )

    hf=figure;
 %   title( sprintf('LR on %s', Exp1(e).dataset) );
    hold on;
    
    fprintf('\n[%s]\n', Exp1(e).dataset );
    fprintf('                                tst AUC          R@90             R@100\n');
    h1 = [];
    str1 = [];
    maxR100=-inf;
    minR50 = inf;
    subplot(1,2,1);
    Tab = [];
    for i = 1 : numel( Exp1(e).Result )
        if exist(Exp1(e).Result(i).fname)
            R = load( Exp1(e).Result(i).fname, 'tstRiskCurve', 'tstAuc' );

            nTst = size( R.tstRiskCurve,1);
            h1(end+1) = plot( [1:nTst]/nTst, mean( R.tstRiskCurve, 2), lineStyle{i}, 'linewidth', 2);
            hold on;
            str1{end+1} = Exp1(e).Result(i).name ;

            th = round( 0.9*nTst);
            Tab(i,:) = [mean(R.tstAuc), std(R.tstAuc), ...
                         mean( R.tstRiskCurve(th,:)),std( R.tstRiskCurve(th,:)),...
                         mean( R.tstRiskCurve(end,:)),std( R.tstRiskCurve(end,:)) ]; 
%             th = round( 0.9*nTst);
%             fprintf('%30s   %.4f(%.4f)  %.4f(%.4f)  %.4f(%.4f)\n', Exp1(e).Result(i).name, ...
%                 mean(R.tstAuc), std(R.tstAuc), ...
%                 mean( R.tstRiskCurve(th,:)),std( R.tstRiskCurve(th,:)),...
%                 mean( R.tstRiskCurve(end,:)),std( R.tstRiskCurve(end,:))); 
            
            maxR100 = max(maxR100, mean( R.tstRiskCurve(end,:)));
            th = round( size(R.tstRiskCurve,1)*0.5);
            minR50  = min(minR50, mean( R.tstRiskCurve(th,:)));
        else
            Tab(i,:) = [nan nan nan nan nan nan];
        end
    end
    for i = 1 : numel( Exp1(e).Result )
        maxAucLeft = ' '; maxAucRight = ' ';
        if Tab(i,1) == min( Tab(:,1) ), maxAucLeft = '['; maxAucRight = ']'; end
        maxR90Left = ' '; maxR90Right = ' ';
        if Tab(i,3) == min( Tab(:,3) ), maxR90Left = '['; maxR90Right = ']'; end
        
        fprintf('%30s %c%.4f(%.4f)%c %c%.4f(%.4f)%c  %.4f(%.4f)\n', Exp1(e).Result(i).name, ...
                maxAucLeft,Tab(i,1),Tab(i,2),maxAucRight, ...
                maxR90Left,Tab(i,3),Tab(i,4),maxR90Right, ...
                Tab(i,5),Tab(i,6));
    end
    
    ha1=gca;
        
    % SVM figure
    hold on;
    
    fprintf('\n[%s]\n', Exp2(e).dataset );
    fprintf('                                tst AUC          R@90             R@100\n');
    h2 = [];
    str2 = [];
    subplot(1,2,2);
    Tab = [];
    for i = 1 : numel( Exp2(e).Result )
        if exist(Exp2(e).Result(i).fname)
            R = load( Exp2(e).Result(i).fname, 'tstRiskCurve', 'tstAuc' );

            nTst = size( R.tstRiskCurve,1);
            h2(end+1) = plot( [1:nTst]/nTst, mean( R.tstRiskCurve, 2), lineStyle{i}, 'linewidth', 2);
            hold on;
            str2{end+1} = Exp2(e).Result(i).name ;

            th = round( 0.9*nTst);
            Tab(i,:) = [mean(R.tstAuc), std(R.tstAuc), ...
                         mean( R.tstRiskCurve(th,:)),std( R.tstRiskCurve(th,:)),...
                         mean( R.tstRiskCurve(end,:)),std( R.tstRiskCurve(end,:)) ]; 
%             th = round( 0.9*nTst);
%             fprintf('%30s   %.4f(%.4f)  %.4f(%.4f)  %.4f(%.4f)\n', Exp2(e).Result(i).name, ...
%                 mean(R.tstAuc), std(R.tstAuc), ...
%                 mean( R.tstRiskCurve(th,:)),std( R.tstRiskCurve(th,:)),...
%                 mean( R.tstRiskCurve(end,:)),std( R.tstRiskCurve(end,:))); 
            
            maxR100 = max(maxR100, mean( R.tstRiskCurve(end,:)));
            th = round( size(R.tstRiskCurve,1)*0.5);
            minR50  = min(minR50, mean( R.tstRiskCurve(th,:)));
        else
            Tab(i,:) = [nan nan nan nan nan nan];
        end
    end
    
    for i = 1 : numel( Exp2(e).Result )
        maxAucLeft = ' '; maxAucRight = ' ';
        if Tab(i,1) == min( Tab(:,1) ), maxAucLeft = '['; maxAucRight = ']'; end
        maxR90Left = ' '; maxR90Right = ' ';
        if Tab(i,3) == min( Tab(:,3) ), maxR90Left = '['; maxR90Right = ']'; end
        
        fprintf('%30s %c%.4f(%.4f)%c %c%.4f(%.4f)%c  %.4f(%.4f)\n', Exp2(e).Result(i).name, ...
                maxAucLeft,Tab(i,1),Tab(i,2),maxAucRight, ...
                maxR90Left,Tab(i,3),Tab(i,4),maxR90Right, ...
                Tab(i,5),Tab(i,6));
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
    print( hf, '-depsc', sprintf('%sLR+SVM_%s.eps', outFolder, Exp1(e).dataset));
    
    drawnow;
    snapnow;
    
end

