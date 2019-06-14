function Data = loss_svorimc_init( X, Y)
% LOSS_SVORIMC_INIT
% 
% Synopsis:
%  Data = loss_svorimc_init( X, Y )
%
% Description:
%

    [Data.nDims,Data.nExamples] = size( X );
    Data.nY = max(Y);

    Data.X = X;
    Data.Y = Y;
    Data.idx = [1:Data.nExamples];

    Data.EY = zeros(Data.nY-1,Data.nExamples);
    for y = 1 : Data.nY-1
        Data.EY(y,:) = 2*double( y < Data.Y(:) ) - 1;
    end

end
% EOF