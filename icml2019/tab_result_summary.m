%% 
% It writes TeX tables to tabs/ folder.
% 

dataSet = {...
    'avila1', ...
    'codrna1',...
    'covtype1',...
    'ijcnn1',...
    'letter1', ...
    'pendigit1',...
    'phishing1',...
    'shuttle1', ...
    'sensorless1',...
    'sattelite1',...
    }

if ~exist('tabs/'), mkdir('tabs'); end

%%
Exp = [];
for i = 1 : numel( dataSet )
    Exp(i).dataset = dataSet{i};
    Exp(i).Result(1).name  = 'LR+plugin';
    Exp(i).Result(1).fname = ['results/lr/' dataSet{i} '/results.mat'];

    Exp(i).Result(end+1).name = 'LR+conf(linear)';
    Exp(i).Result(end).fname  = ['results/lr/' dataSet{i} '/conf_hinge1_linear_zmuv1_th5/results.mat'];
    
    Exp(i).Result(end+1).name  = 'LR+conf(quad)';
    Exp(i).Result(end).fname   = ['results/lr/' dataSet{i} '/conf_hinge1_quad_zmuv1_th5/results.mat'];
    
    Exp(i).Result(end+1).name  = 'LR+conf(mlp)';
    Exp(i).Result(end).fname   = ['results/lr/' dataSet{i} '/conf_hinge1_mlp_zmuv1/results.mat'];

    Exp(i).Result(end+1).name  = 'SVM+maxscore';
    Exp(i).Result(end).fname   = ['results/msvmlin/' dataSet{i} '/results.mat'];

    Exp(i).Result(end+1).name  = 'SVM+conf(linear)';
    Exp(i).Result(end).fname   = ['results/msvmlin/' dataSet{i} '/conf_hinge1_linear_zmuv1_th5/results.mat'];

    Exp(i).Result(end+1).name  = 'SVM+conf(quad)';
    Exp(i).Result(end).fname   = ['results/msvmlin/' dataSet{i} '/conf_hinge1_quad_zmuv1_th5/results.mat'];

    Exp(i).Result(end+1).name  = 'SVM+conf(mlp)';
    Exp(i).Result(end).fname   = ['results/msvmlin/' dataSet{i} '/conf_hinge1_mlp_zmuv1/results.mat'];
        
end



%%
lineStyle = {'r','b','g','m','y','k','c--','k--','b--'};
for e = 1 : numel( Exp )
    figure;
    title( Exp(e).dataset );
    hold on;
    
    fprintf('\n[%s]\n', Exp(e).dataset );
    fprintf('                                  ValLoss(100x) TstLoss(100x)    AUC(100x)     R@90(100x)    R@100(100x)\n');
    h = [];   
    str = [];
    maxR100 = -inf;
    minR50  = inf;
    LrTab.baseAuc = [];  % LR+plugin
    LrTab.baseR90 = [];  % LR+plugin
    LrTab.R100    = [];  % 
    LrTab.auc     = [];  % LR+conf(XXX)
    LrTab.R90     = [];  % LR+conf(XXX)
    LrTab.tstLoss = [];  % LR+conf(XXX)
    LrTab.valLoss = [];  % LR+conf(XXX)
    LrTab.method  = {};
    SvmTab.baseAuc = [];  % LR+plugin
    SvmTab.baseR90 = [];  % LR+plugin
    SvmTab.R100    = [];  % 
    SvmTab.auc     = [];  % LR+conf(XXX)
    SvmTab.R90     = [];  % LR+conf(XXX)
    SvmTab.tstLoss = [];  % LR+conf(XXX)
    SvmTab.valLoss = [];  % LR+conf(XXX)
    SvmTab.method  = {};
    
    
    for i = 1 : numel( Exp(e).Result )
        if exist(Exp(e).Result(i).fname)
            R = load( Exp(e).Result(i).fname, 'tstRiskCurve', 'tstAuc', 'valLoss', 'tstLoss' );

            nTst = size( R.tstRiskCurve,1);
            h(end+1) = plot( [1:nTst]/nTst, mean( R.tstRiskCurve, 2), lineStyle{i}, 'linewidth', 2);
            hold on;
            str{end+1} = Exp(e).Result(i).name ;

            method  = Exp(e).Result(i).name;
            valLoss = 100*[ mean(R.valLoss) std(R.valLoss)];
            tstLoss = 100*[ mean(R.tstLoss) std(R.tstLoss)];
            tstAuc  = 100*[ mean(R.tstAuc) std(R.tstAuc)];
                        
            th90      = round( 0.9*nTst);
            th50      = round( 0.5*nTst);
            R50     = 100*[mean( R.tstRiskCurve(th50,:)) std( R.tstRiskCurve(th50,:))];
            R90     = 100*[mean( R.tstRiskCurve(th90,:)) std( R.tstRiskCurve(th90,:))];
            R100    = 100*[mean(R.tstRiskCurve(end,:)) std( R.tstRiskCurve(end,:))];
                                   
            fprintf('%30s   %5.2f(%5.2f)  %5.2f(%5.2f)  %5.2f(%5.2f)  %5.2f(%5.2f)  %5.2f(%5.2f)\n', ...
                method, ...
                valLoss(1), valLoss(2),...
                tstLoss(1), tstLoss(2),...
                tstAuc(1), tstAuc(2),...
                R90(1), R90(2),...
                R100(1), R100(2));
            
            maxR100 = max(maxR100, R100(1) );
            minR50  = min(minR50, R50(1));
            
            % fill in latextable            
            switch method
                case 'LR+plugin'
                    LrTab.R100 = R100;
                    LrTab.baseAuc = tstAuc;
                    LrTab.baseR90 = R90;
                    LrTab.baseMethod = 'plugin';
                case 'LR+conf(linear)'
                    LrTab.tstLoss(1,:) = tstLoss;
                    LrTab.valLoss(1,:) = valLoss; 
                    LrTab.auc(1,:)     = tstAuc;
                    LrTab.R90(1,:)     = R90;
                    LrTab.method{1}    = 'learn(L)';
                case 'LR+conf(quad)'
                    LrTab.tstLoss(2,:) = tstLoss;
                    LrTab.valLoss(2,:) = valLoss; 
                    LrTab.auc(2,:)     = tstAuc;
                    LrTab.R90(2,:)     = R90;
                    LrTab.method{2}    = 'learn(Q)';
                case 'LR+conf(mlp)'
                    LrTab.tstLoss(3,:) = tstLoss;
                    LrTab.valLoss(3,:) = valLoss; 
                    LrTab.auc(3,:)     = tstAuc;
                    LrTab.R90(3,:)     = R90;
                    LrTab.method{3}    = 'learn(M)';
                    
                case 'SVM+maxscore'
                    SvmTab.R100 = R100;
                    SvmTab.baseAuc = tstAuc;
                    SvmTab.baseR90 = R90;
                    SvmTab.baseMethod = 'maxscore';
                case 'SVM+conf(linear)'
                    SvmTab.tstLoss(1,:) = tstLoss;
                    SvmTab.valLoss(1,:) = valLoss; 
                    SvmTab.auc(1,:)     = tstAuc;
                    SvmTab.R90(1,:)     = R90;
                    SvmTab.method{1}    = 'learn(L)';
                case 'SVM+conf(quad)'
                    SvmTab.tstLoss(2,:) = tstLoss;
                    SvmTab.valLoss(2,:) = valLoss;
                    SvmTab.auc(2,:)     = tstAuc;
                    SvmTab.R90(2,:)     = R90;
                    SvmTab.method{2}    = 'learn(Q)';
                case 'SVM+conf(mlp)'
                    SvmTab.tstLoss(3,:) = tstLoss;
                    SvmTab.valLoss(3,:) = valLoss; 
                    SvmTab.auc(3,:)     = tstAuc;
                    SvmTab.R90(3,:)     = R90;
                    SvmTab.method{3}    = 'learn(M)';
            end            
        end
    end
    
    % figure
    grid on;
    xlabel('colver');
    ylabel('risk');
    legend( h, str );
    a=axis;
    axis([0.5 a(2) minR50/100 maxR100/100]);
    
    % latex tab
    fid = fopen(sprintf('tabs/%s.tex', Exp(e).dataset),'w+');
    
    dataset = upper(Exp(e).dataset(1:end-1));
    dataset = dataset(1:min(length(dataset),7));
    
    fprintf(fid,'\\parbox[t]{2mm}{\\multirow{4}{*}{\\rotatebox[origin=c]{90}{%s}}}\n', dataset );
    [~,idx] = min( LrTab.valLoss(:,1));
    fprintf(fid, '& LR  & %12s & %.1f $\\pm %.1f$ & %.1f $\\pm %.1f$ &  \\multirow{2}{*}{%.1f $\\pm %.1f$} \\\\ \n', ...
        LrTab.baseMethod, LrTab.baseAuc(1), LrTab.baseAuc(2), LrTab.baseR90(1), LrTab.baseR90(2), LrTab.R100(1), LrTab.R100(2));
    fprintf(fid, '& LR  & %12s & %.1f $\\pm %.1f$ & %.1f $\\pm %.1f$  \\\\ \n', ...
        LrTab.method{idx},LrTab.auc(idx,1), LrTab.auc(idx,2), LrTab.R90(idx,1), LrTab.R90(idx,2) );
    
    [~,idx] = min( SvmTab.valLoss(:,1));
    fprintf(fid, '& SVM & %12s & %.1f $\\pm %.1f$ & %.1f $\\pm %.1f$ & \\multirow{2}{*}{%.1f $\\pm %.1f$} \\\\ \n', ...
        SvmTab.baseMethod, SvmTab.baseAuc(1), SvmTab.baseAuc(2), SvmTab.baseR90(1), SvmTab.baseR90(2), SvmTab.R100(1), SvmTab.R100(2));
    fprintf(fid, '& SVM & %12s & %.1f $\\pm %.1f$ & %.1f $\\pm %.1f$ \\\\ \n', ...
        SvmTab.method{idx},SvmTab.auc(idx,1), SvmTab.auc(idx,2), SvmTab.R90(idx,1), SvmTab.R90(idx,2));
    
    fclose(fid);
    
end

