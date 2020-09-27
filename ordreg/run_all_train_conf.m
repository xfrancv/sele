cd /home/xfrancv/Work/SelectiveClassification/sele_journal.github/ordreg;

run('../selclassif_setpath.m');

%dataSet = {{'california1', [100 500 1000 5000 6190] },...
%           {'abalone1', [100 500 1000 1252] },...
%           {'bank1',[100 500 1000 2000 2457]},...
%           {'cpu1',[100 500 1000 2000 2454] }};

dataSet = {{'california1', [100 500 1000 5000 6190] },...
           {'abalone1', [100 500 1000 1252] },...
           {'bank1',[100 500 1000 2000 2457]},...
           {'cpu1',[100 500 1000 2000 2454] },...
           {'msd1',[100 500 1000]}};

    

%% 
for i = 1 : numel( dataSet )
    for trnData = dataSet{i}{2}
        run_train_conf_sele_mlp( dataSet{i}{1}, 'hinge1+zmuv', trnData);    
    end
end

%%
for i = 1 : numel( dataSet )
    for trnData = dataSet{i}{2}
        run_train_conf_regression_mlp( dataSet{i}{1}, 'zmuv', trnData);    
    end
end
       
%% 
for i = 1 : numel( dataSet )
    for trnData = dataSet{i}{2}
        run_train_conf_regression_linear( dataSet{i}{1}, 'zmuv', trnData);    
    end
end

%%
for i = 1 : numel( dataSet )
    for trnData = dataSet{i}{2}
        run_train_conf_regression_quad( dataSet{i}{1}, 'zmuv', trnData);    
    end
end

%% 
for i = 1 : numel( dataSet )
    for trnData = dataSet{i}{2}    
        run_train_conf_sele_linear( dataSet{i}{1}, 'hinge1+zmuv', trnData);  
    end
end

%%
for i = 1 : numel( dataSet )
    for trnData = dataSet{i}{2}    
        run_train_conf_sele_quad( dataSet{i}{1}, 'hinge1+zmuv', trnData);   
    end
end


%%
if ~strcmpi( 'HAL', hostname )
    exit();
end



