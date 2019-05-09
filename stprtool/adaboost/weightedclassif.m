function [predY,score] = weightedclassif( X, Model)
% WEIGHTEDCLASSIF weighted classifier.
%
% Synopsis:
%   predY = weightedclassif( X, Model)
%

% About: Statistical Pattern Recognition Toolbox
% (C) 1999-2014, Written by Vojtech Franc
% <a href="http://www.cvut.cz">Czech Technical University Prague</a>
% <a href="http://www.feld.cvut.cz">Faculty of Electrical Engineering</a>
% <a href="http://cmp.felk.cvut.cz">Center for Machine Perception</a>

    [N,M] = size(X);
    score = zeros(M,1);

    for i = 1 : length( Model.Rule )

      currPredY = Model.Rule{i}.eval( X, Model.Rule{i} );
      currPredY( find(currPredY ~= 1)) = -1;

      score = score + currPredY(:) * Model.alpha(i);
    end

    predY = 2*double( score >= 0) - 1;

return;
% EOF