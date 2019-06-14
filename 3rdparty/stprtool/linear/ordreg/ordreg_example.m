%% Example showing how to learn linear ordinal classifier from examples.
%

load( 'ordreg_data1.mat','X','Y');

%% define loss function to be optimized
nY = max(Y);
Y1 = repmat([1:nY]',1,nY);
Y2 = Y1';
switch 2
    case 1 % 0/1-loss
        lossMatrix = double( Y1 ~= Y2 );

    case 2 % mean absolut error (MAE)
        lossMatrix = abs( Y1 - Y2 );
end


%% init risk computation


%% run a solver
switch 5
    case 1   % quadratic regularizer, batch algorithm
        Opt.verb   = 1;
        Opt.tolAbs = 0.1;
        
        lambda = 0.0001;

        Data  = risk_ordreg_init( X, Y, lossMatrix );
        W     = bmrm( Data, @risk_ordreg, lambda, Opt );
        Model = risk_ordreg_model( Data, W);
        
    case 2   % no regularizer, batch algorithm
        Opt.verb      = 1;
        Opt.tolRel    = 1e-2;
        Opt.boxConstr = 200;

        Data  = risk_ordreg_init( X, Y, lossMatrix );
        W     = ACCPM( Data,@risk_ordreg,[],[],Opt,Opt.tolRel, 0, Opt.boxConstr);
        Model = risk_ordreg_model( Data, W);

    case 3  % quadratic regularizer, online algorithm
        Opt.verb    = 1;
        Opt.tolAbs  = 0.1;
        Opt.bufSize = 1000;

        lambda = 0.0001;
        nExamples = size( X, 2);

        Data  = loss_ordreg_init( X, Y, lossMatrix );
        W     = fasole( Data, nExamples, @loss_ordreg, lambda, Opt );
        Model = loss_ordreg_model( Data, W);

    case 4   % no regularizer, batch algorithm
        Opt.verb      = 1;
        Opt.tolRel    = 1e-2;
        Opt.boxConstr = 200;

        Data  = risk_svorimc_init( X, Y );
        W     = ACCPM( Data,@risk_svorimc,[],[],Opt,Opt.tolRel, 0, Opt.boxConstr);
        Model = risk_svorimc_model( Data, W);

    case 5  % quadratic regularizer, online algorithm
        Opt.verb    = 1;
        Opt.tolAbs  = 0.01;
        Opt.bufSize = 1000;

        lambda = 0.0001;
        nExamples = size( X, 2);

        Data  = loss_svorimc_init( X, Y );
        W     = fasole( Data, nExamples, @loss_svorimc, lambda, Opt );
        Model = loss_svorimc_model( Data, W );
%         Data  = risk_svorimc_init( X, Y );
%         W     = bmrm( Data, @risk_svorimc, lambda, Opt );
%         Model = risk_svorimc_model( Data, W );
end




%% training error
predY  = linclassif( Data.X, Model);
trnerr = sum(predY ~= Data.Y )/length(Data.Y)
trnMae = mean(abs( predY - Data.Y ))

%% display classifier
figure;
ppatterns(Data.X,Data.Y);
axis equal;
hold on;
pclassifier( Model );


