function Data = loss_svorimc_init( X, Y)
% LOSS_SVORIMCIIL_INIT
% 
% Synopsis:
%  Data = loss_svorimciil_init( X, Y )
%
% Description:
%

    [Data.nDims,Data.nExamples] = size( X );
    Data.nY = max( Y(:));

    Data.X = X;
    Data.Y = Y;
    Data.idx = [1:Data.nExamples];

    Data.EY = zeros(Data.nY-1,Data.nExamples);
    for y = 1 : Data.nY-1
        idx            = find(Data.Y(1,:) > y);
        Data.EY(y,idx) = 1;

        idx            = find(Data.Y(1,:) <= y & Data.Y(2,:) > y );
        Data.EY(y,idx) = nan;

        idx            = find( Data.Y(2,:) <= y );
        Data.EY(y,idx) = -1;
    end

end
% EOF