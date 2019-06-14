function [TP,FP,TN,FN,Offset] = roc( score, Y)
% ROC computes Receiver Operating Characteristic (ROC) curve. 
%
% Synopsis:
%  [TP,FP,TN,FN,Offset] = roc( score, Y)
%  
% Description:
%  Consider binary decesion rule predicting according to the sign
%  of a real valued score function, i.e. 
%     if score >= 0 then predict +1
%     if score <  0 then predict -1
%
%  Given the true hidden state Y taking value +1 or -1, the prediction 
%  is evaluated as follows:
%                  True hidden state Y
%                       +1    -1
%  Prediction      +1   TP    FP
%                  -1   FN    TN
%  
%  This function computes the true positive rate TPR and the false
%  positive rate FPR for the prediction rule with its score modified 
%  by an additive offset, i.e.
%    newScore  = score + offset
%  when the bias is varied from min(score)-1 to max(score)+1. 
%   
% Input:
%  score [1 x N] Scores of the prediction rule.
%  Y     [1 x N] True hidden state; +1 is the positive class and
%                anything else stands for the negative class.
%
% Output:
%  TP     [1 x N+1] Number of true positives.
%  FP     [1 x N+1] Number of false positives.
%  TN     [1 x N+1] Number of true negatives.
%  FN     [1 x N+1] Number of false negatives.
%  Offset [1 x N+1] Corresponding offset of the decision score.
%
% Example:
%  % Example 1: plot ROC and precision-recall curve of Fisher classifier
%  load('riply_dataset','Trn','Tst');
%  Model = fld(Trn.X, Trn.Y);
%  [predY,score] = linclassif( Tst.X, Model);
%
%  [TP,FP,TN,FN,Offset] = roc( score, Tst.Y );
%
%  figure; hold on; title('ROC curve');
%  plot( FP./(FP+TN),TP./(TP+FN));
%  xlabel('false positive rate'); 
%  ylabel('true positive rate');
%
%  figure; hold on; title('Precision recall curve');
%  plot( TP./(TP+FN), TP./(TP+FP));
%  xlabel('recall'); 
%  ylabel('precision');
%
%  % Example 2: modify rule to get precision >= 90 with maximal recall
%  [~,idx] = max( -( TP./(TP+FP) < 0.9) + TP./(TP+FN) );
%  newScore = score + Offset( idx );
%  tp = sum( newScore >= 0 & Tst.Y == +1);
%  fp = sum( newScore >= 0 & Tst.Y == -1);
%  fn = sum( newScore <  0 & Tst.Y == +1);
%  prec   = tp/(tp+fp)
%  recall = tp/(tp+fn)
%  
%
% See also 
%

N1 = length(find(Y==1));
N2 = length(find(Y~=1));
N  = N1 + N2;

[sortedScore,inx] = sort(score);
Y                 = Y(inx);

FP = zeros( 1, N+1 );
FN = zeros( 1, N+1 );
TP = zeros( 1, N+1 );
TN = zeros( 1, N+1 );

fn = 0;
fp = N2;
tp = N1;
tn = 0;

for i = 1 : N + 1,
  
    FP(i) = fp;
    FN(i) = fn;
    TP(i) = tp;
    TN(i) = tn;
    
    if i <= N && Y(i) == 1, 
        fn = fn + 1; 
        tp = tp - 1;
    else
        fp = fp - 1; 
        tn = tn + 1;
    end        
end

Offset = -mean( [ sortedScore(:)'   sortedScore(end)+2 ; ...
                  sortedScore(1)-2  sortedScore(:)']);

return;
