function X = d_samp(P,nSamples)
% D_SAMP Generates samples from discrete distribution.
%
% Synopsis:
%  X = dsamp(P,nSamples)
%
% Input:
%  P [nValues x 1]  Discrete probability distribution, i.e.
%                   sum(P) = 1 and min(P) >= 0 must hold.
%  nSamples [1 x 1] Number of samples to be generated.
% Output:
%  X [1 x nSamples] i.i.d. samples from P.
% 
% Example:
%  P = [0.2 0.3 0.1 0.4];
%  X = d_samp(P,10000);
%  P_emp = hist(X,4)/10000
%
% See also

nValues = length(P);
unif_X = rand(1,nSamples);

cumP = repmat(cumsum( P(:) ),1,nSamples);
rnd = ones(nValues,1)*unif_X;
[dummy,X] = max(cumP > rnd,[],1);

return;
% EOF