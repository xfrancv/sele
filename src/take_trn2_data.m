function Data = take_trn2_data( Data, trnData)
%%
    rng(0);

    nY  = max( Data.Y );
    

    for s = 1 : numel( Data.Split )
        if trnData <= 1
            trnDataPercent = trnData*100;
        else
            N = numel( Data.Split(s).trn2 );
            trnDataPercent = min(100*trnData/N,100);
        end

        idx = [];
        for y = 1 : nY
            cidx = Data.Split(s).trn2(find(Data.Y(Data.Split(s).trn2)==y));
            N    = numel( cidx );
            cidx = cidx( randperm( N ));
            M    = round(N*trnDataPercent/100);
            idx  = [idx cidx( 1:M )];
        end

        idx = idx( randperm( numel( idx )));
        Data.Split(s).trn2 = idx;
    end
    
end
