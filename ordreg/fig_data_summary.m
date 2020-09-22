
dataFolder = 'data/';

%%
List = dir( dataFolder );

fprintf('      data   trn1    trn2    val1    val2     tst  #labels #dims\n');
fprintf([ '-'*ones(1,70) '\n']);
for i = 1 : numel( List )
    if ~List(i).isdir
        D = load([dataFolder List(i).name],'Split','Y','X');
        trn1 = [];
        trn2 = [];
        val1 = [];
        val2 = [];
        tst  = [];
        for s = 1 : numel( D.Split)
            trn1(end+1) = numel( D.Split(s).trn1 );
            trn2(end+1) = numel( D.Split(s).trn2 );
            val1(end+1) = numel( D.Split(s).val1 );
            val2(end+1) = numel( D.Split(s).val2 );
            tst(end+1)  = numel( D.Split(s).tst );
        end
        dbName = List(i).name(1:end-5);
        
        fprintf('%10s %6.0f  %6.0f  %6.0f  %6.0f  %6.0f  %5d  %3d\n',...
            dbName, mean(trn1), mean(trn2), mean(val1), ...
            mean(val2),  mean(tst), max(D.Y), size(D.X,1) );
    end
end