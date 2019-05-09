function [trn,val,tst]=tvtsplit(portion,N,K)
% TVTSPLIT Partitions samples to trn/val/tst splits.
%
% Synopsis:
%  [trn,val,tst]=tvtsplit(portion,N,K)
%
% Description:
%  This function generates K random partitions of N samples into 
%  trn/val/tst splits so that the number of trn/val/tst samples 
%  is approximately round(portion*N). 
% 
% Input:
%  portion [1 x 3] portion = [trnPortion valPortion tstPortion];
%                  portion >=0 and sum(portion) = 1;
%  N [1x1] number of samples.
%  K [1x1] number of partitions.
%
% Output:
%  trn [N x K logical] trn(i,j) == 1 indicates that in j-th split 
%          the i-th sample is in training subset.
%  val [N x K logical] val(i,j) == 1 indicates that in in j-th split 
%          the i-th sample is in validation subset.
%  tst [N x K logical] tst(i,j) == 1 indicates that in in j-th split 
%          the i-th sample is in testing subset.
%
% Example:
%   [trn,val,tst]=tvtsplit([0.7 0.1 0.2],10,5)
%

    % random partitioning 
    trn = zeros( N,K,'logical');
    val = zeros( N,K,'logical');
    tst = zeros( N,K,'logical');

    th = round( N * portion);
    
    for k = 1 : K
        idx = randperm( N );
        
        trn(idx(1:th(1)),k)             = 1;
        val(idx(th(1)+1:th(1)+th(2)),k) = 1;
        tst(idx(th(1)+th(2)+1:N),k)     = 1;
    end

end
% EOF
