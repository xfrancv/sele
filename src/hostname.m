function str=hostname()
% str=hostname()
    [~,str] = system('hostname');
    str(find(double(str)<32)) = [];
end
