rng(0);

switch 2
    case 1
        outFile  = '../../../data/msd1.mat';
        portions = [0.3 0.3 0.1 0.1 0.2]; % trn1 trn2 val1 val2 tst
        nSplits  = 5;
        skipLabels = 1;
    case 2
        outFile  = '../../../data/msd2.mat';
        portions = [0.3 0.3 0.1 0.1 0.2]; % trn1 trn2 val1 val2 tst
        nSplits  = 5;
        skipLabels = 0;
end

%%
% if ~exist( outFile )
%     system('wget --no-check-certificate https://www.dcc.fc.up.pt/~ltorgo/Regression/bank32NH.tgz');
%     system('tar -xvzf bank32NH.tgz');  
% end


%%
A = dlmread( 'YearPredictionMSD.txt',',');
X = A(:,2:end)';
Y = A(:,1);


%% 
if skipLabels
    idx = find( Y >= 1970 & Y <=2010);
    X = X(:,idx);
    Y = Y(idx);
end

[~,~,Y] = unique( Y );
nY = max(Y);

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
