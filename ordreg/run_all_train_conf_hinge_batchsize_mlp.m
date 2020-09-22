run('../selclassif_setpath.m');

dataSet = {{'california1', [50 100 200 500]},...
           {'abalone1', [50 100 200 500]},...
           {'bank1',[50 100 200 500]},...
           {'cpu1',[50 100 200 500]}};,...
           %{'msd1',[100 500 1000 2000]}};
           

%% 
for i = 1 : numel( dataSet )
    for batchSize = dataSet{i}{2}
        run_train_conf_hinge_mlp( dataSet{i}{1}, 'hinge1+zmuv', batchSize);    
    end
end





%%
if ~strcmpi( 'HAL', hostname )
    exit();
end



