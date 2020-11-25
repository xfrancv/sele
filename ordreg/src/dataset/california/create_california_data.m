rng(0);

switch 2
    case 1
        outFile  = '../../../data/california1.mat';
        portions = [0.3 0.3 0.1 0.1 0.2]; % trn1 trn2 val1 val2 tst
        nSplits  = 5;
        nY = 10;
    case 2
        outFile  = '../../../data/california2.mat';
        portions = [0.3 0.3 0.1 0.1 0.2]; % trn1 trn2 val1 val2 tst
        nSplits  = 5;
        nY = 100;
        
end

%%
% if ~exist( outFile )
%     system('wget --no-check-certificate https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.tgz');
%     system('tar -xvzf cal_housing.tgz');
% end


%%
[x1,x2,x3,x4,x5,x6,x7,x8,y] = textread('CaliforniaHousing/cal_housing.data', '%f%f%f%f%f%f%f%f%f', 'delimiter',',');

X = [x1 x2 x3 x4 x5 x6 x7 x8]';
Y = y(:);


%%
% target value, Y, is discretized into nY ordinal quantities using
% equal-frequency binning
[N,E] = histcounts( Y, linspace( min(Y), max(Y)+1e-3,1000));
C = cumsum( N );

th = E(1);
for i = 1:nY
    a = max(find( C <= i*C(end)/nY));
    th = [th E(a+1)];
end

newY = Y;
for i = 1 : nY
    idx = find( Y >= th(i) & Y < th(i+1));
    newY(idx) = i;
end

Y = newY;

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
