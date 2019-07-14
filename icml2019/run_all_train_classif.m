run('../selclassif_setpath.m');

dataSet = {'avila1', 'codrna1', 'letter1', 'pendigit1', 'shuttle1',...
          'sattelite1','covtype1', 'sensorless1', 'phishing1','ijcnn1'};
       

for i = 1 : numel( dataSet )
    run_train_msvmlin( dataSet{i}, 'zmuv+reg0.1-100' );
end

for i = 1 : numel( dataSet )
    run_train_lr( dataSet{i}, 'zmuv+reg0-100' );
end

if ~strcmpi( 'X1-VF', hostname )
    exit();
end
