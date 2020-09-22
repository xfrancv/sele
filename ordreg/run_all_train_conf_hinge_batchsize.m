run('../selclassif_setpath.m');

dataSet = {{'california1', [100 500 1000 2000]},...
           {'abalone1', [100 500 1000 2000]},...
           {'bank1',[100 500 1000 2000]},...
           {'cpu1',[100 500 1000 2000]},...
           {'msd1',[100 500 1000 2000]}};
           

%% 
for i = 1 : numel( dataSet )
    for batchSize = dataSet{i}{2}
        run_train_conf_hinge_linear( dataSet{i}{1}, 'hinge1+zmuv', batchSize);    
    end
end

%% 
for i = 1 : numel( dataSet )
    for batchSize = dataSet{i}{2}
        run_train_conf_hinge_quad( dataSet{i}{1}, 'hinge1+zmuv', batchSize);    
    end
end





%%
if ~strcmpi( 'HAL', hostname )
    exit();
end



