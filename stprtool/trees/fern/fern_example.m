% Example: Train FERN classifier on Riply's dataset. Nodes of weak trees 
%   are decision stumps generated randomly.
% 


load('riply_dataset','Trn','Tst');

% transform labels (+1,-1) -> (1,2)
Trn.Y(find(Trn.Y~=1)) = 2;
Tst.Y(find(Tst.Y~=1)) = 2;

% setting
nTrees     = 100;
treeHeight = 4;
gridDens   = 100;
Min        = min( Trn.X, [], 2);
Max        = max( Trn.X, [], 2);


% train FERN
Model = fern_train( Trn.X, Trn.Y, nTrees, treeHeight, ...
                    @() stump_tree_create(treeHeight, Min, Max, gridDens),...
                    @stump_tree_eval );

% training error
trnPredY = fern_classif( Trn.X, Model );
trnErr   = sum( trnPredY(:) ~= Trn.Y(:)) / length(Trn.Y)

% testing error
tstPredY = fern_classif( Tst.X, Model );
tstErr   = sum( tstPredY(:) ~= Tst.Y(:)) / length( Tst.Y )

% display decision boundary
figure; 
ppatterns( Trn.X,Trn.Y);
pclassifier(Model, @fern_classif);
