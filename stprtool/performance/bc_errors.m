function E = bc_errors( score, Y )
% BC_ERRORS evaluates performance of a binary classifier
%
% Synopsis:
%  E = bc_errors( score, Y )
%
% Description:
%   The binary classifier predicts POSITIVE class if score >= 0 and NEGATIVE 
%   class otherwise. The label Y==1 means POSITIVE class and Y~=1 means 
%   NEGATIVE class.
%
%   The function computes:
%
%   # of true negatives     E.tn  = sum( trueY == -1 & predY == -1 )
%   # of true positives     E.tp  = sum( trueY == +1 & predY == +1 )
%   # of false negatives    E.fn  = sum( trueY == +1 & predY == -1 )
%   # of false positives    E.fp  = sum( trueY == -1 & predY == +1 )
%
%   classification error    E.clserr   = ( fn + fp ) / (tp + fn + tn + fp )
%   accuracy                E.accuracy = 1 - E.clserr
%   false positive rate     E.fpr      = fp  / (fp + tn )
%   false negative rate     E.fnr      = fn  / (fn + tp )
%   precision               E.prec     = tp  / (tp + fp )
%   recall                  E.recall   = tp  / (tp + fn )
%   F1 score                E.F1       = 2*tp / (2*tp + fp + fn )
%   Area under ROC          E.auc
%


    trueY = 2*double( Y(:) == 1 )     - 1;
    predY = 2*double( score(:) >= 0 ) - 1;

    nFalseNeg = sum( trueY == +1 & predY == -1 );
    nFalsePos = sum( trueY == -1 & predY == +1 );
    nTruePos  = sum( trueY == +1 & predY == +1 );
    nTrueNeg  = sum( trueY == -1 & predY == -1 );

    E.clserr   = ( nFalseNeg+nFalsePos ) / ( nFalsePos+nTruePos+nFalseNeg+nTrueNeg );
    E.accuracy = 1 - E.clserr;
    E.fpr      = nFalsePos  / (nFalsePos+nTrueNeg );
    E.fnr      = nFalseNeg  / (nFalseNeg+nTruePos );
    E.fn       = nFalseNeg;
    E.fp       = nFalsePos;
    E.tp       = nTruePos;
    E.tn       = nTrueNeg;
    E.prec     = nTruePos   / (nTruePos + nFalsePos );
    E.recall   = nTruePos   / (nTruePos + nFalseNeg );
    E.F1       = 2*nTruePos / (2*nTruePos + nFalsePos + nFalseNeg );
    E.auc      = rocarea( score(:), Y(:) );

end