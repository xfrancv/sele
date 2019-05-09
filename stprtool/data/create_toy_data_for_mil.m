%% Create toy data for MIL from completely labeled exampels
% which can be clicked by createdata('finite',xx);
%
% y in "raw" data created by "createdata('finite',20)"
% y =  1 ... negative bags of size 2
% y >= 2 ... positive bags
%


switch 2
    case 1
        inFile  = 'toy_data_mil1_raw.mat';
        outFile = 'toy_data_mil1.mat';
    case 2
        inFile  = 'toy_data_mil2_raw.mat';
        outFile = 'toy_data_mil2.mat';
end

%%
load(inFile,'X','y');
Z = y;

nZ        = max( Z );
bagIdx    = [];
bagLabels = [];
Y         = nan*ones(length(Z),1);

cnt = 0;
for z = 1 : nZ
    if z == 1
        idx = find( Z == z );
        Y(idx) = -1;
        for i = idx(:)'
            cnt = cnt + 1;
            bagIdx{cnt}    = i;
            bagLabels(cnt) = -1;
        end
    else        
        idx            = find( Z == z );
        Y(idx)         = 1;
        cnt            = cnt + 1;
        bagIdx{cnt}    = idx;
        bagLabels(cnt) = 1;
    end
end

save( outFile, 'X','Y','Z','bagLabels','bagIdx' );

