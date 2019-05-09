% Example: Latent SVM with known latent labels.
%

lambda = 0.001;  % regularization
X0     = 1;      % bias

OrigData = load( 'fourclassproblem.mat','Y','X');
yzmap    = {[1 4],[2 3]}; 

Data     = risk_latentsvm_init( OrigData.X, OrigData.Y, yzmap, X0 );
        

%% rename classes
for y = 1 : Data.nY 
    for z = Data.yzmap{y}
        OrigData.Y( find( Data.Z == z) ) = y;
    end
end
 
%%
Opt.verb = 1;
[W,Stat] = bmrm( Data, @risk_latentsvm, lambda, Opt );

Model    = risk_latentsvm_model( W, Data );

predY    = linclassif( OrigData.X, Model );
trnErr   = sum(predY(:) ~= OrigData.Y(:)) / length( predY );

%%
figure;
ppatterns( OrigData.X, OrigData.Y,'bigcircles');
ppatterns( OrigData.X, Data.Z,'labels');    
pclassifier( Model, @linclassif );

