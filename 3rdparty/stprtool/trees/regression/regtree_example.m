%% Example: usage of regression tree
%

%% create training data and plot them
M          = 100;
trnX       = linspace(0,2*pi,M);
noiseLessY = sin(trnX)';
trnY       = noiseLessY + 0.2*randn(size(noiseLessY));

figure;
plot( trnX, trnY, 'xk' ); hold on;
plot( trnX, noiseLessY, 'k');


%% Example 1: fit regression tree

Opt = [];
Opt.minLeafExamples = 5; 
Opt.minSseImprov    = 1e-6;
Opt.maxDepth        = 5; 

Tree   = regtree_train( trnX, trnY, Opt );
predY  = regtree_pred( trnX, Tree );

trnMse = mean( (predY - trnY).^2 )

plot( trnX, predY, 'g');



%% Example 2: show effect of the tree depth

figure;
plot( trnX, trnY, 'xk' ); hold on;
plot( trnX, noiseLessY, 'k');

h   = [];
str = [];
for depth = [1, 5, 10]
    Opt.maxDepth = depth;
    Tree         = regtree_train( trnX, trnY, Opt );
    predY        = regtree_pred( trnX, Tree );
    trnMse       = mean( (predY - trnY).^2 );

    h(end+1)   = plot( trnX, predY, 'linewidth',2 );
    str{end+1} = sprintf('depth=%2d, mse=%.4f', depth, trnMse);   
end
legend( h, str );

