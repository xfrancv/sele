function [y,score]=linclassif( X, model)
% LINCLASSIF Linear classifier.
%
% Synopsis:
%  [y,score] = linclassif( X, model )
%
% Description:
%  If length(W0) > 1 then this function implements multi-class linear
%  classifier:
%
%    y(i) = argmax W(:,y)'*X(:,i) + W0(y)    i=1:nVectors
%              y
%
%  where parameters W [nDim x nY] and W0 [nY x 1] are encapsulated in 
%  the structure "model" and X [nDim x nVectors]. If the maximum is
%  attained for more labels then the classifier returns the minimal one.  
%
%  If length(W0) == 1 then the function implements two-class linear
%  classifier:
%
%    y(i) = +1 if W'*X(:,i) + W0 >= 0      i=1:nVectors
%           -1 if W'*X(:,i) + W0 < 0
%  
% Input:
%  X [nDim x nVectors] Vectors to be classified.
%  model [struct] Parameters of the linear classifier (see above).
%
% Output:
%  y [nVectors x 1] Predicted labels.
%  score [nRules x nVectors] Score function (see above).
%
% Examples:
%  load('riply_dataset', 'Trn', 'Tst' );
%  Model  = fld( Trn.X, Trn.Y );
%  ypred  = linclassif( Trn.X, Model );
%  trnErr = mean( ypred(:) ~= Trn.Y(:))
%  ypred  = linclassif( Tst.X, Model );
%  tstErr = mean( ypred(:) ~= Tst.Y(:))
%  figure; 
%  ppatterns( Trn.X, Trn.Y ); 
%  pline( Model.W, Model.W0 );
%
% See also 
%

%%
[nDim, nVectors] = size( X );
nRules = size(model.W,2);

%%
if nRules == 1,
  % two-class classifier
  score = model.W'*X + model.W0;
  y = ones(nVectors,1);
  y(score < 0) = -1;

else   
  % multi-class classifier
  score = model.W'*X + model.W0(:)*ones(1,nVectors);
  [dummy,y] = max( score );
end

if isfield( model, 'map' )

    origY = y;
    for i = 1 : numel( model.map )
        for z = model.map{i}
            idx = find( z == origY );
            y(idx) = i;
        end
    end

end

return;
