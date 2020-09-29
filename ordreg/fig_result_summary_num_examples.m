%

outFolder = 'figs/';

dataSet = {{'california1', [100 500 1000 5000 6190] },...
           {'abalone1', [100 500 1000 1252] },...
           {'bank1',[100 500 1000 2000 2457]},...
           {'cpu1',[100 500 1000 2000 2454] },...
           {'msd1',[100 500 1000 5000 10000]}};


%methodSuffix = {'_linear_zmuv1','_quad_zmuv1','_mlp_zmuv1'};
methodSuffix = {'_linear_zmuv1','_mlp_zmuv1'};

%
if ~exist(outFolder ), mkdir( outFolder ); end


for e = 1 : numel( dataSet )

    Exp1  = [];

    fprintf('%s\n', dataSet{e}{1});
    hf=figure('name', dataSet{e}{1} );

    for m=1:numel(methodSuffix)


        expName = methodSuffix{m};
        idx = find( expName == '_');
        expName = expName(idx(1)+1:idx(2)-1);
        %fprintf('[[%s]]\n', expName);
        %drawnow;
        %snapnow;


        Exp1.dataset = dataSet{e}{1};
        Exp1.Result(1).name = sprintf('SVORIMC+regression(%s)', expName);
        Exp1.Result(1).fnamePrefix  = sprintf('results/svorimc/%s/conf_regression%s', dataSet{e}{1}, methodSuffix{m});
        Exp1.Result(1).trnData = dataSet{e}{2};
        Exp1.Result(2).name = sprintf('LR+sele(%s)', expName);
        Exp1.Result(2).fnamePrefix  = sprintf('results/svorimc/%s/conf_sele1%s',dataSet{e}{1}, methodSuffix{m} );
        Exp1.Result(2).trnData = dataSet{e}{2};    


        %
        lineStyle = {'k','r','g','b','m'};
    
        h1 = [];
        str1 = [];
        subplot(1,numel(methodSuffix),m);
        for i = 1 : numel( Exp1.Result )

            tstAuc = [];
            for d= Exp1.Result(i).trnData
                fname = sprintf('%s_trn%d/results.mat', Exp1.Result(i).fnamePrefix, d);
                R = load( fname, 'tstAuc' );
                tstAuc = [tstAuc mean(R.tstAuc)];
            end

            str1{end+1} = Exp1.Result(i).name;
            h1(end+1) = semilogx( Exp1.Result(i).trnData, tstAuc, lineStyle{i}, 'linewidth', 2);
            hold on;
        end
        ha1=gca;
        title(Exp1.dataset);

        axes(ha1);
        grid on;
        xlabel('#trn examples');
        ylabel('auRC');
        legend( h1, str1,'Location','northeast' );
        h=gca;
        h.FontSize=15;


    end
    set(hf,'Position', [709 718 1811 620]);
    drawnow;
    snapnow;
end
%close all;
