%% Example: It plots prec-recall curve for varying prior probability of classes
% 

%% Load data, train Fisher classifier, compute test prediction
load('riply_dataset','Trn','Tst');
Model = fld(Trn.X, Trn.Y);
[predY,score] = linclassif( Tst.X, Model);

%% compute errors as a function of varying decision threshold
[TP,FP,TN,FN,Offset] = roc( score, Tst.Y );

%% display PRC curve for varying P(y==+1)
figure; hold on; title('Precision recall curve');
xlabel('recall'); 
ylabel('precision');

Npos = sum( Tst.Y == 1 );
Nneg = sum( Tst.Y == 1 );

txt = [];
h   = [];

for py = [0.01 0.25 0.5 0.75 0.99]


    prec   = (py*TP/Npos)./( (py*TP/Npos)+(1-py)*FN/Nneg );
    recall = TP./(TP+FP);
    
    h(end+1)   = plot( prec, recall,'linewidth', 2,'linestyle','--');
    txt{end+1} = sprintf('P(y=+1)=%.2f', py);
    
    if py == Npos/(Npos+Nneg), set( h(end), 'linestyle','-'); end
end
 
legend( h, txt, 'Location', 'southwest' );
