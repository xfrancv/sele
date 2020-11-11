run('../selclassif_setpath.m');

dataSet = {'avila1', 'codrna1','covtype1','ijcnn1','letter1',...
           'pendigit1', 'phishing1', 'sattelite1','sensorless1','shuttle1' };

       
dataSet = {{'avila1', [100 500 1000 5000 6258]}, ...
           {'codrna1',[100 500 1000 5000 10000 15000 16557]}, ...
           {'covtype1',[100 500 1000 5000 10000 11620]}, ...
           {'ijcnn1',[100 500 1000 5000 10000 14000 14997]},...
           {'letter1',[100 500 1000 5000 5999]},...
           {'pendigit1',[100 500 1000 3000 3295]},...
           {'phishing1',[100 500 1000 3000 3317]},...
           {'sattelite1',[100 500 1000 1500 1932]},...
           {'sensorless1',[100 500 1000 5000 10000 15000 17545]},...
           {'shuttle1',[100 500 1000 5000 10000 15000 17401]} };
    
           
S = [];
%% linear/quad/mlp conf rule trained on LR
for i = 1 : numel( dataSet )
    for trnData = dataSet{i}{2}
%        S{end+1} = run_train_conf_logistic_linear( dataSet{i}{1}, 'lr+zmuv', trnData);    
        S{end+1} = run_train_conf_logistic_quad( dataSet{i}{1}, 'lr+zmuv', trnData);    
%        S{end+1} = run_train_conf_logistic_mlp( dataSet{i}{1}, 'lr+zmuv', trnData);    

%        S{end+1} = run_train_conf_logistic_linear( dataSet{i}{1}, 'msvmlin+zmuv', trnData);    
        S{end+1} = run_train_conf_logistic_quad( dataSet{i}{1}, 'msvmlin+zmuv', trnData);    
%        S{end+1} = run_train_conf_logistic_mlp( dataSet{i}{1}, 'msvmlin+zmuv', trnData);    
    end
end

%% linear/quad/mlp conf rule trained on LR
for i = 1 : numel( dataSet )
    for trnData = dataSet{i}{2}
%        S{end+1} = run_train_conf_sele_linear( dataSet{i}{1}, 'lr+sele1+zmuv', trnData);    
        S{end+1} = run_train_conf_sele_linear( dataSet{i}{1}, 'lr+sele2+zmuv', trnData);    
        S{end+1} = run_train_conf_sele_linear( dataSet{i}{1}, 'lr+sele3+zmuv', trnData);    

        S{end+1} = run_train_conf_sele_quad( dataSet{i}{1}, 'lr+sele1+zmuv', trnData);    
%        S{end+1} = run_train_conf_sele_mlp( dataSet{i}{1}, 'lr+sele1+zmuv', trnData);    
%        S{end+1} = run_train_conf_sele_mlp( dataSet{i}{1}, 'lr+sele2+zmuv', trnData);    

%        S{end+1} = run_train_conf_sele_linear( dataSet{i}{1}, 'msvmlin+sele1+zmuv', trnData);    
        S{end+1} = run_train_conf_sele_linear( dataSet{i}{1}, 'msvmlin+sele2+zmuv', trnData);    
        S{end+1} = run_train_conf_sele_linear( dataSet{i}{1}, 'msvmlin+sele3+zmuv', trnData);    

        S{end+1} = run_train_conf_sele_quad( dataSet{i}{1}, 'msvmlin+sele1+zmuv', trnData);    
%        S{end+1} = run_train_conf_sele_mlp( dataSet{i}{1}, 'msvmlin+sele1+zmuv', trnData);    
%        S{end+1} = run_train_conf_sele_mlp( dataSet{i}{1}, 'msvmlin+sele2+zmuv', trnData);    
    end
end

%
idx = find( arrayfun( @(x) ~isempty(x{1}), S) );
for i = idx
    fprintf('%s\n', S{i});
end


%%
if ~strcmpi( 'HAL', hostname )
    exit();
end
