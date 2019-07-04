run('../selclassif_setpath.m');;

% dataSet = {'avila1', 'codrna1','covtype1','ijcnn1','letter1',...
%            'pendigit1', 'phishing1', 'sattelite1','sensorless1','shuttle1' };
dataSet = {'ijcnn1' };

    
%% linear conf rule trained on LR 
for i = 1 : numel( dataSet )
    run_train_conf_hinge_linear( dataSet{i}, 'lr+hinge1+zmuv+par5');    
end

%% linear conf rule trained on SVM 
for i = 1 : numel( dataSet )
    run_train_conf_hinge_linear( dataSet{i}, 'msvmlin+hinge1+zmuv+par5' );    
end

%% quad conf rule train on LR
for i = 1 : numel( dataSet )
    run_train_conf_hinge_quad( dataSet{i}, 'lr+hinge1+zmuv+par5' );
end

%% quad conf rule train on SVM
for i = 1 : numel( dataSet )
    run_train_conf_hinge_quad( dataSet{i}, 'msvmlin+hinge1+zmuv+par5' );
end

%% MLP conf rule trained on LR
for i = 1 : numel( dataSet )
    run_train_conf_hinge_mlp( dataSet{i}, 'lr+hinge1+zmuv' );
end

%% MLP conf rule trained on SVM
for i = 1 : numel( dataSet )
    run_train_conf_hinge_mlp( dataSet{i}, 'msvmlin+hinge1+zmuv' );
end

%%
if ~strcmpi( 't430s-vf', hostname )
    exit();
end



