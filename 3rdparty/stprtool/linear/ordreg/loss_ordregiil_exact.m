function val = loss_ordregiil_exact(Data, W )
% LOSS_ORDREGIIL_EXACT compute exact loss.
%
%  vals = loss_ordregiil_exact(Data, W )
%  val  = loss_ordregiil_exact(i, Data, W )
%

    Model = loss_ordregiil_model( Data, W );
    
    if nargin == 3
        idx = Data.idx(i);
    else
        idx = Data.idx;
    end
            
    predY  = linclassif( Data.X(:,idx), Model); predY = predY(:);
    trueYl = Data.Y(1,idx)';
    trueYr = Data.Y(2,idx)';

    N = length( predY );
    val = zeros(1,N);

    
    for y = 1 : Data.nY
        idx = find( predY < trueYl & trueYl == y );
        if ~isempty(idx), val(idx) = Data.lossMatrix( y, predY(idx) ); end

        idx = find( predY > trueYr & trueYr == y ); 
        if ~isempty(idx), val(idx) = Data.lossMatrix( y, predY(idx) ); end
    end
end
