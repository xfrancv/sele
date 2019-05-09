function [proxy, phi, loss] = loss_pwmord( i, Data, W )
% LOSS_PWMORD Piece-wise Ordinal classifier.
% 
% Synopsis:
%  [proxy, phi, loss] = loss_pwmord( i, Data )
%  [proxy, phi, loss] = loss_pwmord( i, Data, W )
%
% Description:
%

    j = Data.idx( i );

    if nargin < 3, W = zeros( Data.nDims*Data.nZ + Data.nY, 1 ); end

    V0 = W( Data.nDims*Data.nZ+1:end);
    
    %V  = reshape( W(1:Data.nDims*Data.nZ), Data.nDims, Data.nZ );
    %Q  = V'*Data.X(:,j); 

%     P  = V*Data.A;    
%     score = P' * Data.X(:,j) + V0;
%    P  = V*Data.A;    
    Q = zeros(Data.nZ,1);
    for k = 1 : Data.nZ, 
        Q(k) = W((k-1)*Data.nDims+1:k*Data.nDims)'*Data.X(:,j);
    end
    score = Data.A'*Q + V0;
    
    
    delta = abs( [1:Data.nY]' - Data.Y(j) );
    score = delta + score;

%    score = score - P(:,Data.Y(j))'*Data.X(:,j) - V0(Data.Y(j));
    score = score - Data.A(:,Data.Y(j))'*Q - V0(Data.Y(j));

    [proxy, ypred] = max( score );
    
    loss  = abs( ypred - Data.Y( j ) );
    
    phi1  = Data.X(:,j)*sparse(Data.A(:,ypred) - Data.A(:,Data.Y(j)))';
    
    phi2            = zeros(Data.nY,1);
    phi2(ypred)     = 1;
    phi2(Data.Y(j)) = phi2(Data.Y(j)) - 1;
    
    phi = [phi1(:) ; phi2(:)];

end
