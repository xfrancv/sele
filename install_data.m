dataSet = {'codrna','avila', 'letter', 'pendigit', 'shuttle',...
           'sattelite','covtype', 'sensorless','phishing','ijcnn1'};

if ~exist('data/')
    mkdir('data/');
end
       
for i = 1 : numel( dataSet )
    cd(['datasets/' dataSet{i} '/']);
    run(['create_' dataSet{i} '_data.m']);
    cd ../../;
end
