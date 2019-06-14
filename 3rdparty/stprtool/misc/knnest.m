% KNNEST K-Nearest Neighbour estimator.
%
% Synopsis:
%  out = knnest( tst_X, trn_X, trn_y, K )
%  out = knnest( tst_X, trn_X, trn_y )
%
% Description:
%  This function finds K nearest neigbours for each testing example. The 
%  output out(y,i) is the number of training examples trn_X with label 
%  trn_y==y found among K nearest neighbors of the testing example 
%  tst_X(:,i).
%
%  K-NN classifier is obtained by
%    [dummy, predicted_labels] = max( out )
%
%  K-NN estimate of the posterior probability P( label | feature ) is 
%  obtained by
%    posterior = out/K
%   
% Input:
%  tst_X [nDim x nTst] test feature vectors.
%  trn_X [nDim x nTrn] training feature vectors.
%  trn_y [ nTrn x 1] training labels; must be integers from 1 to nY.
%  K [1 x 1] number of the nearest neighbours (default 1).
%
% Output:
%  out [nY x nTst] out(y,i) is the number of training features with 
%    label y found among K nearest neighbors of the vector tst_X(:,i).
%
% Examples:
%  load('riply_dataset','Trn','Tst');
%  K = 3;
%  [dummy, predy] = max( knnest(Tst.X, Trn.X, Trn.Y, K) );
%  test_error = sum(predy ~= Tst.Y)/length(Tst.Y)
%  
