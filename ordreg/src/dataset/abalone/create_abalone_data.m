rng(0);

outFile  = '../../../data/abalone1.mat';
portions = [0.3 0.3 0.1 0.1 0.2]; % trn1 trn2 val1 val2 tst
nSplits  = 5;

%%
% if ~exist( outFile )
%     system('wget --no-check-certificate https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data');    
%     system('wget --no-check-certificate https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.names');
% end


%%
[x1,x2,x3,x4,x5,x6,x7,x8,y] = textread('abalone.data', '%c%f%f%f%f%f%f%f%f', 'delimiter',',');



N = numel(x1);
x10 = zeros(N,1);
x20 = zeros(N,1);
x30 = zeros(N,1);
x10( find( x1=='F')) = 1;
x20( find( x1=='I')) = 1;
x30( find( x1=='M')) = 1;


X = [x10 x20 x30 x2 x3 x4 x5 x6 x7 x8]';

%%
% merge labels 1-3   and 21-29
Y = y(:);
for k = [1:2]
    Y(find(y==k)) = 3;
end
for k = [22:29]
    Y(find(y==k)) = 21;
end
[~,~,y] = unique(Y);

Y = y;


nY = max(y);

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
    fprintf('      trn1  trn2  val1  val2   tst\n');
    for y = 1 : nY
        fprintf('y=%2d ',y);
        fprintf('%5d ', sum( Y( Split(s).trn1 ) == y));
        fprintf('%5d ', sum( Y( Split(s).trn2 ) == y));
        fprintf('%5d ', sum( Y( Split(s).val1 ) == y));
        fprintf('%5d ', sum( Y( Split(s).val2 ) == y));
        fprintf('%5d ', sum( Y( Split(s).tst ) == y));
        fprintf('\n');
    end
    fprintf('     %5d %5d %5d %5d %5d\n', numel(Split(s).trn1), ...
        numel(Split(s).trn2), numel(Split(s).val1), numel(Split(s).val2),...
        numel(Split(s).tst));
end

%%
Y = Y(:)';
save( outFile, 'X', 'Y', 'Split', '-v7.3'  );
% EOF
