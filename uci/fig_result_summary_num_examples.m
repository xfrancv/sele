%

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

methodSuffix = {'_linear_zmuv1','_mlp_zmuv1'};

%
if ~exist(outFolder ), mkdir( outFolder ); end


for m=1:numel(methodSuffix)


    expName = methodSuffix{m};
    idx = find( expName == '_');
    expName = expName(idx(1)+1:idx(2)-1);
    fprintf('[[%s]]\n', expName);
    drawnow;
    snapnow;

    Exp1  = [];
    Exp2 = [];


    for e = 1 : numel( dataSet )
        Exp1(e).dataset = dataSet{e}{1};
        Exp1(e).Result(1).name = sprintf('LR+logistic(%s)', expName);
        Exp1(e).Result(1).fnamePrefix  = sprintf('results/lr/%s/conf_logistic%s', dataSet{e}{1}, methodSuffix{m});
        Exp1(e).Result(1).trnData = dataSet{e}{2};
        Exp1(e).Result(2).name = sprintf('LR+sele(%s)', expName);
        Exp1(e).Result(2).fnamePrefix  = sprintf('results/lr/%s/conf_sele1%s',dataSet{e}{1}, methodSuffix{m} );
        Exp1(e).Result(2).trnData = dataSet{e}{2};

        Exp2(e).dataset = dataSet{e}{1};
        Exp2(e).Result(1).name = sprintf('SVM+logistic(%s)', expName);
        Exp2(e).Result(1).fnamePrefix  = sprintf('results/msvmlin/%s/conf_logistic%s', dataSet{e}{1}, methodSuffix{m});
        Exp2(e).Result(1).trnData = dataSet{e}{2};
        Exp2(e).Result(2).name = sprintf('SVM+sele(%s)', expName);
        Exp2(e).Result(2).fnamePrefix  = sprintf('results/msvmlin/%s/conf_sele1%s', dataSet{e}{1}, methodSuffix{m} );
        Exp2(e).Result(2).trnData = dataSet{e}{2};
    end



    %
    lineStyle = {'k','r','g','b','m'};
    k = 0.3;
    shadeColor = { [1 1 1]-k, [1 k k], [k 1 k], [k k 1]};
    for e = 1 : numel( Exp1 )

        hf=figure('name', Exp1(e).dataset );
%        fprintf('%s\n', Exp1(e).dataset);
        snapnow;
        
        
        h1 = [];
        str1 = [];
        subplot(1,2,1);
        for i = 1 : numel( Exp1(e).Result )

            tstAuc = [];
            for d= Exp1(e).Result(i).trnData
                fname = sprintf('%s_trn%d/results.mat', Exp1(e).Result(i).fnamePrefix, d);
                R = load( fname, 'tstAuc' );
                tstAuc = [tstAuc mean(R.tstAuc)];
            end

            str1{end+1} = Exp1(e).Result(i).name;
            h1(end+1) = semilogx( Exp1(e).Result(i).trnData, tstAuc, lineStyle{i}, 'linewidth', 2);
            hold on;
        end
        ha1=gca;
        title(Exp1(e).dataset);


        h2 = [];
        str2 = [];
        subplot(1,2,2);
        for i = 1 : numel( Exp2(e).Result )

            tstAuc = [];
            for d= Exp2(e).Result(i).trnData
                fname = sprintf('%s_trn%d/results.mat', Exp2(e).Result(i).fnamePrefix, d );
                if exist(fname)
                    R = load( fname, 'tstAuc' );
                    tstAuc = [tstAuc mean(R.tstAuc)];
                else
                    tstAuc = [tstAuc nan];
                end
            end

            str2{end+1} = Exp2(e).Result(i).name;
            h2(end+1) = semilogx( Exp2(e).Result(i).trnData, tstAuc, lineStyle{i}, 'linewidth', 2);
            hold on;
        end
        ha2=gca;

        axes(ha1);
        grid on;
        xlabel('#trn examples');
        ylabel('auRC');
        legend( h1, str1,'Location','northeast' );
        h=gca;
        h.FontSize=15;

        axes(ha2);
        grid on;
        xlabel('#trn examples');
        ylabel('auRC');
        legend( h2, str2,'Location','northeast' );
        h=gca;
        h.FontSize=15;
        hf.Position = [520 374 993 424];
        ha1.Position=[0.1300 0.1800 0.3347 0.75];
        ha2.Position=[0.5703 0.1800 0.3347 0.750];
    %    print( hf, '-depsc', sprintf('%sLR+SVM_%s.eps', outFolder, Exp1(e).dataset));
        drawnow;
        snapnow;

    end
end
