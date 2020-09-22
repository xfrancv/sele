run('../selclassif_setpath.m');

dataSet = {{'california1', [100 500 1000 5000 6000],1000,1000},...
           {'abalone1', [100 500 1000 1200], 100,100},...
           {'bank1',[100 500 1000 1254],500,1000 },...
           {'cpu1',[100 500 1000 2000 2459],2000,1000 }};

    
           
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
        run_train_conf_hinge_linear( dataSet{i}{1}, 'hinge1+zmuv', dataSet{i}{3}, trnData);    
    end
end

%%
for i = 1 : numel( dataSet )
    for trnData = dataSet{i}{2}    
        run_train_conf_hinge_quad( dataSet{i}{1}, 'hinge1+zmuv', dataSet{i}{4}, trnData);    
    end
end



%%
if ~strcmpi( 'HAL', hostname )
    exit();
end



