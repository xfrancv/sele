function auc = rocarea( score, labels )
% ROCAREA computes area under ROC curve.
% 
% Synopsis:
%    val = rocarea( score, labels )
%
% Input:
%   score  [N x 1] real valued scores of two-class classifier.
%   labels [N x 1] labels; 1.. class 1 and enything else for class 2
%
% Output:
%   val [1 x 1] area under ROC curve.
%
% Example:
%  load('riply_dataset','Trn','Tst');
%  Model = fld(Trn.X, Trn.Y);
%  [predY,score] = linclassif( Tst.X, Model);
%
%  rocarea( score, Tst.Y ) 
% 
%  [TP,FP,TN,FN] = roc( score, Tst.Y );
%  figure; hold on; title('roc curve');
%  plot( FP./(FP+TN),TP./(TP+FN));
%  xlabel('false positive rate'); 
%  ylabel('true positive rate');
%

    N = length( score );        

    [sortedScore,idx] = sort( score );
    sortedLabels = labels( idx );

    pos = 0;
    neg = 0;
    auc = 0;
    for i = 1 : N
        if sortedLabels(i) == 1
            pos = pos + 1;
        else
            neg = neg + 1;
            auc = auc + pos;
        end
    end

    auc = 1 - auc / (neg*pos);

end
