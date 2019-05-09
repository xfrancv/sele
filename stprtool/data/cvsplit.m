function [trn,tst]=cvsplit(N,K)
% CVSPLIT Partitions samples for K-fold cross-validation.
%
% Synopsis:
%  [trn,tst]=cvsplit(N,K)
%
% Description:
%  This function randomly partitions N samples into K-folders 
%  and assignes them to training and testing sub-sets. 
% 
% Input:
%  N [1x1] number of samples.
%  K [1x1] number of folders.
%
% Output:
%  trn [N x K logical] trn(i,j) == 1 indicates that in j-th split that 
%      i-th sample is in training subset.
%  tst [N x K logical] tst(i,j) == 1 indicates that in in j-th split that 
%      i-th sample is in testing subset.
%

    % random partitioning 
    idx = randperm( N );

    trn = zeros( N,K,'logical');
    tst = zeros( N,K,'logical');
    
    numOfColumns = ceil( N/K );
    part = [1:N zeros(1,numOfColumns*K-N)];
    part = reshape(part, numOfColumns,K)';

    for k = 1 : K
        tstIdx = part(k,:);
        tstIdx = tstIdx(find(tstIdx));        
        trnIdx = part(find([1:K]~=k),:);
        trnIdx = trnIdx(find(trnIdx));

        trn(idx(trnIdx),k) = 1;
        tst(idx(tstIdx),k) = 1;
    end

end
% EOF
