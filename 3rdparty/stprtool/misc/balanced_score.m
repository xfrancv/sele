function biasCorrection = balanced_score(score,labels)
% BALANCED_SCORE Compute balancing bias.
%
% Synopsis:
%   biasCorrection = balanced_score(score,labels)
%
% Description:
%   Let score [n x 1] contains values of scoring function of a binary
%   classifier and let labels [n x 1] be corresponding labels. 
%   This function computes balancing bies, i.e.
%     balancedScore = score + balanced_score(score,labels)
%   such that the error on the positive and the negative class are equal.
%     
% Example:
%   load('riply_dataset','Trn');
%   [W,W0,stat] = svmocas( Trn.X,1, Trn.Y,10);
%   score = W'*Trn.X + W0;
%
%   errPosClass = sum(sign(score(:))==-1 & Trn.Y(:) == +1)/sum(Trn.Y== +1)
%   errNegClass = sum(sign(score(:))==+1 & Trn.Y(:) == -1)/sum(Trn.Y== -1)
%
%   biasCorrection = balanced_score(score, Trn.Y)
%   score = score + biasCorrection;
%   errPosClass = sum(sign(score(:))==-1 & Trn.Y(:) == +1)/sum(Trn.Y== +1)
%   errNegClass = sum(sign(score(:))==+1 & Trn.Y(:) == -1)/sum(Trn.Y== -1)
% 

    score  = score(:);
    labels = labels(:);

    % compute balanced error
    [sortedScore,idx] = sort(score);
    sortedLabels      = labels(idx);
    posErrors         = [0 cumsum(sortedLabels==+1)'];
    negErrors         = [1+sum(sortedLabels==-1)-cumsum(sortedLabels==-1)' 0];
    [dummy,idx]       = min(abs(posErrors-negErrors));
    if idx > 1 & idx < length( sortedScore )
        biasCorrection = -0.5*(sortedScore(idx-1) + sortedScore(idx));
    else
        biasCorrection = 0;
    end

return;