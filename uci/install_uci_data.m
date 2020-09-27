run('../selclassif_setpath');


dataSet = {'codrna','avila', 'letter', 'pendigit', 'shuttle',...
           'sattelite','covtype', 'sensorless','phishing','ijcnn1'};

       
if ~exist('data/')
    mkdir('data/');
end

for i = 1 : numel( dataSet )
    cd(['src/datasets/' dataSet{i} '/']);
    run(['create_' dataSet{i} '_data.m']);
    cd ../../../;
end


% remove working files
delete src/datasets/codrna/cod-rna;
delete src/datasets/codrna/cod-rna.t;
delete src/datasets/covtype/covtype.data;
delete src/datasets/ijcnn1/ijcnn1.svmlight;
delete src/datasets/ijcnn1/ijcnn1.tr;
delete src/datasets/ijcnn1/ijcnn1.val;
delete src/datasets/letter/letter-recognition.data;
delete src/datasets/pendigit/pendigits.tes;
delete src/datasets/pendigit/pendigits.tra;
delete src/datasets/phishing/phishing;
delete src/datasets/sattelite/sat.trn;
delete src/datasets/sattelite/sat.tst;
delete src/datasets/sensorless/Sensorless;
delete src/datasets/shuttle/shuttle.trn;
delete src/datasets/shuttle/shuttle.tst;
delete src/datasets/avila/avila/avila-description.txt;
delete src/datasets/avila/avila/avila-tr.txt;
delete src/datasets/avila/avila/avila-ts.txt;
rmdir('src/datasets/avila/avila/');
