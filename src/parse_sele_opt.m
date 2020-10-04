function [benchmark,riskType,zmuvNorm] = parse_sele_opt(setting)
%%

    idx = find( setting == '+');
    
    if length( idx ) == 2
        zmuvNorm = 1;
        numStr = setting( idx(2)-1);
    else
        numStr = setting( end);
    end
    
    riskType = str2num( numStr );
    benchmark = setting(1:idx(1)-1);
    
end