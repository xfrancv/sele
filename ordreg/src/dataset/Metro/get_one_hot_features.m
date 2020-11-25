function feat = get_one_hot_features( X )
%

    [~,~,lab] = unique( X );
    
    nX = max(lab);
    N = numel(lab);
    feat = zeros(nX, N);

    feat(double(lab)+[0:N-1]'*nX) = 1;
end