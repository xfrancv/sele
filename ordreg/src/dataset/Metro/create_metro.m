%%
rng(0);

switch 1
    case 1
        outFile  = '../../../data/metro1.mat';
        portions = [0.3 0.3 0.1 0.1 0.2]; % trn1 trn2 val1 val2 tst
        nSplits  = 5;
        nY = 10;
    case 2
        outFile  = '../../../data/metro2.mat';
        portions = [0.3 0.3 0.1 0.1 0.2]; % trn1 trn2 val1 val2 tst
        nSplits  = 5;
        nY = 100;
end

%%
[holiday,temp,rain_1h,snow_1h,clouds_all,weather_main,weather_descr,date_time, traffic_volume] = ...
    textread('Metro_Interstate_Traffic_Volume.csv', ...
    '%s%f%f%f%f%s%s%s%f','delimiter',',','headerlines',1);


%
day=arrayfun(@(x) x{1}(6:10), date_time, 'UniformOutput', false);
hour=arrayfun(@(x) x{1}(12:13), date_time, 'UniformOutput', false);

X = [get_one_hot_features( holiday ) ; ...
     temp(:)' ; ...
     rain_1h(:)'; ...
     snow_1h(:)'; ...
     clouds_all(:)'; ...
     get_one_hot_features( weather_main ); ...
     get_one_hot_features( weather_descr ); ...
     get_numeric_features( day ); ...
     get_numeric_features( hour )];
 
Y = traffic_volume;

%%
% target value, Y, is discretized into nY ordinal quantities using
% equal-frequency binning
[N,E] = histcounts( Y, linspace( min(Y), max(Y)+1e-3,2000));
C = cumsum( N );

th = E(1);
for i = 1:nY
    a = max(find( C <= C(1)+i*(C(end)-C(1))/nY));
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


