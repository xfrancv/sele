%% Example showing how to train Piece-Wise Ordinal Classifier.
%

load('s10','X','Y');

cutLabels = [1 4 6 10];  % define pieces of ordered labels, e.g. like here 1-4, 4-6, 6-10

switch 2
    case 1
        Opt.verb    = 1;
        Opt.tolAbs  = 0.1;
        Opt.bufSize = 500;
        Opt.maxIter = 1000;

        lambda = 0.0001;        
        nExamples = length( Y );
        
        Data   = loss_pwmord_init( X, Y, cutLabels );
        W      = fasole( Data, nExamples, @loss_pwmord, lambda, Opt );
        Model  = loss_pwmord_model( Data, W );
    
    case 2
        Opt.verb   = 1;
        Opt.tolRel = 1e-2;
        boxConstr  = 100;
        
        Data   = risk_pwmord_init( X, Y, cutLabels );
        W      = ACCPM( Data,@risk_pwmord,[],[],Opt,Opt.tolRel, 0, boxConstr);
        Model  = risk_pwmord_model( Data, W );
end

predY  = linclassif( Data.X, Model );
trnerr = sum(predY(:) ~= Data.Y(:) )/length(Data.Y)
trnMae = mean(abs( predY(:) - Data.Y(:) ))

figure;
axis([-1 1 -1 1]);
ppatterns( Data.X, Data.Y,'labels');
hold on;
pclassifier( Model );
