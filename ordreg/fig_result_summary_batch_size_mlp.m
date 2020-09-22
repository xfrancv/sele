%

outFolder = 'figs/';

dataSet = {{'california1', [50 100 200 500]},...
    {'abalone1', [50 100 200 500]},...
    {'bank1',[50 100 200 500]},...
    {'cpu1',[50 100 200 500 ]},...
    {'msd1',[50 100 200 500]}};

% dataSet = {...
%      'codrna1'}; 
%  ,...
%      'avila1'}; 
%  , ...
%     'codrna1',...
%     'covtype1',...
%     'ijcnn1',...
%     'letter1', ...
%     'pendigit1',...
%     'phishing1',...
%     'sattelite1',...
%     'sensorless1',...
%     'shuttle1', ...
%     }

%methodSuffix1 = {'_linear_zmuv1','_quad_zmuv1'};
methodSuffix1 = {'_mlp_zmuv1'};

%
if ~exist(outFolder ), mkdir( outFolder ); end


for m=1:numel(methodSuffix1)


    expName = methodSuffix1{m};
    idx = find( expName == '_');
    expName = expName(idx(1)+1:idx(2)-1);
    fprintf('[[%s]]\n', expName);
    drawnow;
    snapnow;

    Exp1  = [];
    Exp2 = [];


    for e = 1 : numel( dataSet )
        Exp1(e).dataset = dataSet{e}{1};
        Exp1(e).Result(1).name = sprintf('LR+sele(%s)', expName);
        Exp1(e).Result(1).fnamePrefix  = sprintf('results/svorimc/%s/conf_hinge1%s',dataSet{e}{1}, methodSuffix1{m} );
        Exp1(e).Result(1).batchSize = dataSet{e}{2};
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
%        subplot(1,2,1);
        for i = 1 : numel( Exp1(e).Result )

            tstAuc = [];
            for d= Exp1(e).Result(i).batchSize
                fname = sprintf('%s_bs%d/results.mat', Exp1(e).Result(i).fnamePrefix, d);
                if exist(fname)
                    R = load( fname, 'tstAuc' );
                end
                tstAuc = [tstAuc mean(R.tstAuc)];
            end

            str1{end+1} = Exp1(e).Result(i).name;
            h1(end+1) = semilogx( Exp1(e).Result(i).batchSize, tstAuc, lineStyle{i}, 'linewidth', 2);
            hold on;
        end
        ha1=gca;
        title(Exp1(e).dataset);


        axes(ha1);
        grid on;
        xlabel('cover');
        ylabel('risk');
        legend( h1, str1,'Location','northeast' );
        h=gca;
        h.FontSize=15;

        drawnow;
        snapnow;

    end
end
