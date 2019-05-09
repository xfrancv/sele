%% Example showing how to learn linear ordinal classifier from examples.
%

load( 'ordreg_data1.mat','X','Y');


%%
nExamples = length(Y);
nY = max(Y);

%% define loss function to be optimized
Y1 = repmat([1:nY]',1,nY);
Y2 = Y1';
switch 2
    case 1 % 0/1-loss
        lossMatrix = double( Y1 ~= Y2 );

    case 2 % mean absolut error (MAE)
        lossMatrix = abs( Y1 - Y2 );
end

%% generate imprecisely labeled data
%labMap    = [1 3; 4 6; 7 10];       
labMap    = [1 2; 3 4; 5 6; 7 8; 9 10];       
intervalY = nan*ones(2,nExamples);
for i = 1 : size( labMap,1 )
    idx = find( labMap(i,1) <= Y & labMap(i,2) >= Y);
    intervalY(1,idx) = labMap(i,1);
    intervalY(2,idx) = labMap(i,2);
end

% use portion of th precisely labeled examples
th  = round( nExamples*0.1);
idx = randperm( nExamples);
intervalY(1,idx(1:th)) = Y(idx(1:th));
intervalY(2,idx(1:th)) = Y(idx(1:th));


%% quadratic regularizer, online algorithm
Opt.verb    = 1;
Opt.tolAbs  = 0.01;
Opt.bufSize = 1000;

lambda = 0.0001;
nExamples = size( X, 2);

%
Data1  = loss_ordreg_init( X, Y, lossMatrix );
W1     = fasole( Data1, nExamples, @loss_ordreg, lambda, Opt );
Model1 = loss_ordreg_model( Data1, W1);

%
Data2  = loss_ordregiil_init( X, intervalY, lossMatrix );
W2     = fasole( Data2, nExamples, @loss_ordregiil, lambda, Opt );
Model2 = loss_ordregiil_model( Data2, W2);

%
Data3  = loss_svorimciil_init( X, intervalY );
W3     = fasole( Data3, nExamples, @loss_svorimciil, lambda, Opt );
Model3 = loss_svorimciil_model( Data3, W3);


%% training error
predY1  = linclassif( X, Model1);
trnerr1 = sum(predY1 ~= Y )/length(Y)
trnMae1 = mean(abs( predY1 - Y ))


predY2     = linclassif( X, Model2);
trnerr2    = sum(predY2 ~= Y )/length(Y)
trnMae2    = mean(abs( predY2 - Y ))
trnMaeIil2 = mean(loss_ordregiil_exact( Data2, W2 ))

predY3     = linclassif( X, Model3);
trnerr3    = sum(predY3 ~= Y )/length(Y)
trnMae3    = mean(abs( predY3 - Y ))
trnMaeIil3 = mean(loss_svorimciil_exact( Data3, W3 ))

%% display classifier
figure;
ppatterns(Data1.X,Data1.Y);
axis equal;
hold on;
pclassifier( Model1 );
pclassifier( Model2,[],struct('LineSpec','r') );
pclassifier( Model3,[],struct('LineSpec','m') );


