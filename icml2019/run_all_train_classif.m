setpath_selclassif();

% dataSet = {'codrna1','avila1', 'letter1', 'pendigit1', 'shuttle1',...
%            'sattelite1','covtype1', 'sensorless1','phishing1','ijcnn1'};
dataSet = {'codrna1','avila1', 'letter1', 'pendigit1', 'shuttle1',...
           'sattelite1','covtype1', 'phishing1','ijcnn1'};

for i = 1 : numel( dataSet )
    run_train_msvmlin( dataSet{i}, 'zmuv+reg0.1-100' );
end

for i = 1 : numel( dataSet )
    run_train_lr( dataSet{i}, 'zmuv+reg0-100' );
end

if ~strcmpi( 't430s-vf', hostname )
    exit();
end
    