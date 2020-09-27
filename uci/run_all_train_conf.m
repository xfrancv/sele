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
    
           
%% linear/quad/mlp conf rule trained on LR
for i = 1 : numel( dataSet )
    for trnData = dataSet{i}{2}
        run_train_conf_logistic_linear( dataSet{i}{1}, 'lr+zmuv', trnData);    
        run_train_conf_logistic_quad( dataSet{i}{1}, 'lr+zmuv', trnData);    
        run_train_conf_logistic_mlp( dataSet{i}{1}, 'lr+zmuv', trnData);    
    end
end

%% linear/quad/mlp conf rule trained on MSVMLIN
for i = 1 : numel( dataSet )
    for trnData = dataSet{i}{2}
        run_train_conf_logistic_linear( dataSet{i}{1}, 'msvmlin+zmuv', trnData);    
        run_train_conf_logistic_quad( dataSet{i}{1}, 'msvmlin+zmuv', trnData);    
        run_train_conf_logistic_mlp( dataSet{i}{1}, 'msvmlin+zmuv', trnData);    
    end
end

%% linear/quad/mlp conf rule trained on LR
for i = 1 : numel( dataSet )
    for trnData = dataSet{i}{2}
        run_train_conf_sele_linear( dataSet{i}{1}, 'lr+hinge1+zmuv', trnData);    
        run_train_conf_sele_quad( dataSet{i}{1}, 'lr+hinge1+zmuv', trnData);    
        run_train_conf_sele_mlp( dataSet{i}{1}, 'lr+hinge1+zmuv', trnData);    
    end
end

%% linear/quad/mlp conf rule trained on MSVMLIN
for i = 1 : numel( dataSet )
    for trnData = dataSet{i}{2}
        run_train_conf_sele_linear( dataSet{i}{1}, 'msvmlin+hinge1+zmuv', trnData);    
        run_train_conf_sele_quad( dataSet{i}{1}, 'msvmlin+hinge1+zmuv', trnData);    
        run_train_conf_sele_mlp( dataSet{i}{1}, 'msvmlin+hinge1+zmuv', trnData);    
    end
end

%%
if ~strcmpi( 'HAL', hostname )
    exit();
end



