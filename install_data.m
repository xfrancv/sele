setpath_selclassif;


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


% remove working files
delete datasets/codrna/cod-rna;
delete datasets/codrna/cod-rna.t;
delete datasets/covtype/covtype.data;
delete datasets/ijcnn1/ijcnn1.svmlight;
delete datasets/ijcnn1/ijcnn1.tr;
delete datasets/ijcnn1/ijcnn1.val;
delete datasets/letter/letter-recognition.data;
delete datasets/pendigit/pendigits.tes;
delete datasets/pendigit/pendigits.tra;
delete datasets/phishing/phishing;
delete datasets/sattelite/sat.trn;
delete datasets/sattelite/sat.tst;
delete datasets/sensorless/Sensorless;
delete datasets/shuttle/shuttle.trn;
delete datasets/shuttle/shuttle.tst;
delete datasets/avila/avila/avila-description.txt;
delete datasets/avila/avila/avila-tr.txt;
delete datasets/avila/avila/avila-ts.txt;
rmdir('datasets/avila/avila/');
