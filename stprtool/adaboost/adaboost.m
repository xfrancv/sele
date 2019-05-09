function Model = adaboost(X, Y, weakLearner, Opt, varargin)
% ADABOOST AdaBoost algorithm.
%
% Synopsis:
%  Model = adaboost(X, Y, weakLearner)
%  Model = adaboost(X, Y, weakLearner, Opt)
%  Model = adaboost(X, Y, weakLearner, Opt, varargin)
%
% Description:
%  This function implements the AdaBoost algorithm which
%  produces a classifier composed from a set of weak rules.
%  The weak rules are learned by a weakLearner. The task 
%  of the weak learner is to produce a rule with weighted 
%  error less then 0.5. The Adaboost algorithm calls in 
%  each stage the weak learner
%
%     Rule = weakLearner(X,Y,weights)
%     
%  where weights [M x 1] is a vector assigning a weight
%  from 0 to 1 to each training example. If varargin are nonempty
%  then the call Rule = weakLearner(X,Y,weights, varargin) is used
%  instaed. The Rule produced by a weakLearner must contain a handle 
%  of function that implements a weak classifier, i.e. calling
%
%     predY = Rule.eval( X, Rule )
%
%  classifis examples in X.
%
% Input:
%   X [N x M] Training vectors.
%   Y [M x 1] Labels of training vectos (1 and anything else).
%   weakLearner [function] Weak learner; see above.
%
%  Options [struct] :
%   .maxIter   [1x1] Maximal number of iterations = weak rules (default 100).
%   .targetErr [1x1] Halt if training error reaches targetErr (default 0.001).
%   .verb      [1x1] If 1 then displayed progress info.
%
% Output:
%  Model [struct] AdaBoost classifier:
%   .rule    [cell array T x 1] Weak classification rules.
%   .alpha   [T x 1] Weights of the weak rules.
%   .werr    [T x 1] Weighted errors of the weak rules.
%   .Z       [T x 1] Normalization constants of the distribution D.
%   .trnErr  [T x 1] Training error.
%
% Example:
%   load('riply_dataset','Trn');
%
%   Model = adaboost( Trn.X, Trn.Y, @decision_stump );
% 
%   figure; 
%   ppatterns( Trn.X, Trn.Y); 
%   pclassifier( Model, @weightedclassif );
% 
%   figure; hold on; 
%   h1 = plot( Model.werr, 'g');
%   h2 = plot( Model.trnErr, 'b');
%   legend([h1 h2],'weighted error','training error');
%
% See also: 
%  ADABOOSTCLASSIF.
%

% About: Statistical Pattern Recognition Toolbox
% (C) 1999-2014, Written by Vojtech Franc and Vaclav Hlavac
% <a href="http://www.cvut.cz">Czech Technical University Prague</a>
% <a href="http://www.feld.cvut.cz">Faculty of Electrical Engineering</a>
% <a href="http://cmp.felk.cvut.cz">Center for Machine Perception</a>

    %
    if nargin < 4, Opt = []; end
    if ~isfield(Opt,'maxIter'),   Opt.maxIter  = 100; end
    if ~isfield(Opt,'targetErr'), Opt.targetErr = 0.001; end
    if ~isfield(Opt,'verb'),      Opt.verb     = 1; end

    % take data dimensions
    [N,M] = size( X );

    % convert labels to +1 and -1
    Y(find(Y~=1)) = -1;

    % initial distribution over training samples
    weights = ones( M, 1 ) / M;

    %
    Model.alpha    = [];
    Model.Z        = [];
    Model.werr     = [];
    Model.trnErr   = [];
    Model.eval     = @weightedclassif;

    % score of combined (strong) rule
    score = zeros(M,1);  

    %
    nIter = 0;
    go    = true;
    while go

      nIter = nIter + 1;

      if Opt.verb, fprintf('iter %d: ', nIter); end

      % call weak learner
      if numel(varargin) > 0
        Rule = weakLearner( X, Y, weights, varargin );
      else
        Rule = weakLearner( X, Y, weights );
      end

      % 
      predY = Rule.eval( X, Rule );
      predY(find(predY~=1)) = -1;

      werr   = (predY(:) ~= Y(:) )'* weights(:);
      if Opt.verb, fprintf('werr=%f', werr); end

      if werr < 0.5,

        alpha = 0.5*log((1-werr)/werr);

        yh = 2*(predY(:) == Y(:) )-1;
        D  = weights .* exp( -alpha*yh(:));

        % normalize the weights
        Z       = sum( D );
        weights = D / Z;

        % upper bound on the training error
        score    = score + alpha*predY(:);
        predY    = 2*(score>=0) - 1;
        trnErr   = sum( predY(:) ~= Y(:)) / M;

        % store variables
        Model.Z(nIter)        = Z;
        Model.alpha(nIter)    = alpha;
        Model.Rule{nIter}     = Rule;
        Model.trnErr( nIter ) = trnErr;

        % stopping conditions
        if nIter >= Opt.maxIter
            go  = false;
            Model.exitstatus = 'Maximal number of iterations achieved';
        elseif trnErr <= Opt.targetErr
            go  = false;
            Model.exitStatus = 'Target training error achieved.';
        end

        if Opt.verb, fprintf(', alpha=%f, trnErr=%f\n', alpha, trnErr ); end

      else
        % the weighted error is over 0.5
        if Opt.verb, fprintf('>= 0.5 thus stop.\n'); end

        go = false;
        Model.exitStatus = 'Weighted error of weak rule >= 0.5.';
      end

      Model.werr(nIter) = werr;

    end

return;
% EOF