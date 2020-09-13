run('../selclassif_setpath.m');;

dataSet = {{'avila1', [100 500 1000 5000]}, ...
           {'codrna1',[100 500 1000 5000 10000 15000]}, ...
           {'covtype1',[100 500 1000 5000 10000]}, ...
           {'ijcnn1',[100 500 1000 5000 10000 15000]},...
           {'letter1',[100 500 1000 5000]},...
           {'pendigit1',[100 500 1000 3000]},...
           {'phishing1',[100 500 1000 3000]},...
           {'sattelite1',[100 500 1000 1500]},...
           {'sensorless1',[100 500 1000 5000 10000 15000]},...
           {'shuttle1',[100 500 1000 5000 10000 15000]} };
    
           
%% linear conf rule trained on LR using logistic regression loss
for i = 1 : numel( dataSet )
    for trnData = dataSet{i}{2}
        run_train_conf_logistic_linear( dataSet{i}{1}, 'lr+zmuv', trnData);    
    end
end

%% linear conf rule trained on SVM using logistic regression loss
for i = 1 : numel( dataSet )
    for trnData = dataSet{i}{2}
        run_train_conf_logistic_linear( dataSet{i}, 'msvmlin+zmuv',trnData);    
    end
end

%% linear conf rule trained on LR
for i = 1 : numel( dataSet )
    for trnData = dataSet{i}{2}    
        run_train_conf_hinge_linear( dataSet{i}, 'lr+hinge1+zmuv+par5', trnData);    
    end
end

%% linear conf rule trained on SVM 
for i = 1 : numel( dataSet )
    for trnData = dataSet{i}{2}
        run_train_conf_hinge_linear( dataSet{i}, 'msvmlin+hinge1+zmuv+par5', trnData );    
    end
end

%%
if ~strcmpi( 'HAL', hostname )
    exit();
end



