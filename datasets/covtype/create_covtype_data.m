rng(0);

outFile = '../../data/covtype1.mat';
portions = [0.28 0.02 0.2 0.2 0.3]; % trn1 trn2 val1 val2 tst
nSplits  = 5;

%%
if ~exist( outFile )
    gunzip('http://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz');
end

%%
A = dlmread( 'covtype.data', ',' );
X = A(:,1:end-1)';
Y = A(:,end);

%%
[labels,~,Y] = unique( Y );
nY = max( Y );

%%
Split = [];
field = {'trn1','trn2','val1','val2','tst'};

for s = 1 : nSplits
    for i = 1 : numel( field )
        Split(s).(field{i}) = [];
    end
    
    for y = 1 : nY
        idx = find( Y(:)' == y );
        N   = numel( idx );
        idx = idx( randperm( N ) );
        
        th = [0 round( cumsum( portions )*N)];
        
        for i = 1 : numel( field )
            Split(s).(field{i}) = [Split(s).(field{i}) idx(th(i)+1:th(i+1))];
        end
    end
    
    for i = 1 : numel( field )
        idx = Split(s).(field{i});
        Split(s).(field{i}) = idx( randperm( numel(idx)));
    end
end

%%
for s = 1 : nSplits
    fprintf('[split=%d]\n', s);
    fprintf(' trn1  trn2  val1  val2   tst\n');
    for y = 1 : nY
        fprintf('%5d ', sum( Y( Split(s).trn1 ) == y));
        fprintf('%5d ', sum( Y( Split(s).trn2 ) == y));
        fprintf('%5d ', sum( Y( Split(s).val1 ) == y));
        fprintf('%5d ', sum( Y( Split(s).val2 ) == y));
        fprintf('%5d ', sum( Y( Split(s).tst ) == y));
        fprintf('\n');
    end
end

%%
Y = Y(:)';
save( outFile, 'X', 'Y', 'Split', 'labels', '-v7.3'  );
% EOF
