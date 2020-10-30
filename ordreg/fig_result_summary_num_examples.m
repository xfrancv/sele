%

outFolder = 'figs/';

dataSet = {{'california1', [100 500 1000 5000 6190] },...
           {'abalone1', [100 500 1000 1252] },...
           {'bank1',[100 500 1000 2000 2457]},...
           {'cpu1',[100 500 1000 2000 2454] },...
           {'msd1',[100 500 1000 5000 10000]}, ...
           {'bikeshare1', [100 500 1000 5213] },...
           {'ccpp1', [100 500 1000 2872] },...
           {'facebook1',[100 500 1000 5000 10000]},...
           {'gpu1',[100 500 1000 5000 10000] },...
           {'superconduct1',[100 500 1000 5000 6378]}};


%methodSuffix = {'_linear_zmuv1','_quad_zmuv1','_mlp_zmuv1'};
methodSuffix = {'_linear_zmuv1','_mlp_zmuv1'};

%
if ~exist(outFolder ), mkdir( outFolder ); end


for e = 1 : numel( dataSet )

    Exp  = [];

    fprintf('%s\n', dataSet{e}{1});
    hf=figure('name', dataSet{e}{1} );


    %fprintf('[[%s]]\n', expName);
    %drawnow;
    %snapnow;


    Exp.dataset = dataSet{e}{1};
    Exp.Result(1).name = sprintf('reg(lin)');
    Exp.Result(1).fnamePrefix  = sprintf('results/svorimc/%s/conf_regression_linear_zmuv1', dataSet{e}{1});
    Exp.Result(1).trnData = dataSet{e}{2};
    Exp.Result(2).name = sprintf('sele1(lin)');
    Exp.Result(2).fnamePrefix  = sprintf('results/svorimc/%s/conf_sele1_linear_zmuv1',dataSet{e}{1} );
    Exp.Result(2).trnData = dataSet{e}{2};    
    Exp.Result(3).name = sprintf('reg(mlp)');
    Exp.Result(3).fnamePrefix  = sprintf('results/svorimc/%s/conf_regression_mlp_zmuv1', dataSet{e}{1});
    Exp.Result(3).trnData = dataSet{e}{2};
    Exp.Result(4).name = sprintf('sele1(mlp)');
    Exp.Result(4).fnamePrefix  = sprintf('results/svorimc/%s/conf_sele1_mlp_zmuv1', dataSet{e}{1});
    Exp.Result(4).trnData = dataSet{e}{2};


    %
    lineStyle = {'k','r','g','b','m'};

    h1 = [];
    str1 = [];
    for i = 1 : numel( Exp.Result )

        tstAuc = [];
        for d= Exp.Result(i).trnData
            fname = sprintf('%s_trn%d/results.mat', Exp.Result(i).fnamePrefix, d);
            R = load( fname, 'tstAuc' );
            tstAuc = [tstAuc mean(R.tstAuc)];
        end

        str1{end+1} = Exp.Result(i).name;
        h1(end+1) = semilogx( Exp.Result(i).trnData, tstAuc, lineStyle{i}, 'linewidth', 2);
        hold on;
    end
    ha1=gca;
    title(Exp.dataset);

    axes(ha1);
    grid on;
    xlabel('#trn examples');
    ylabel('auRC');
    legend( h1, str1,'Location','northeast' );
    h=gca;
    h.FontSize=15;


    set(hf,'Position', [746 630 860 645]);
    drawnow;
    snapnow;
end
%close all;
