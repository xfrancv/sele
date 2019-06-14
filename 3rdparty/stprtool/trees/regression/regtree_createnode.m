function [Node,numChildren] = regtree_createnode( trnX, trnY, depth, Opt )
% 
%   [Node,numChildren] = regtree_createnode( trnX, trnY, Opt )
%

    
    [N,M] = size( trnX );
    global leafNodeCnt;

    Node.value       = mean( trnY );
    Node.trnExamples = M;
    Node.depth       = depth;
    
    if M <= Opt.minLeafExamples || depth >= Opt.maxDepth
        
        leafNodeCnt = leafNodeCnt + 1;
        Node.isLeaf = 1;
        Node.id     = leafNodeCnt;
        numChildren = 1;

        if M <= Opt.minLeafExamples
            Node.stop = sprintf('#examples %d <= minLeafExamples %d', M, Opt.minLeafExamples);
            if isnan( Node.value )
                disp( Node );
                error('Node value is NAN.');
            end
        else
             Node.stop = sprintf('Maximal depth %d reached.', Opt.maxDepth );
        end
    else
        
        th  = zeros(N,1);
        sse = zeros(N,1);
        for i = 1 : N
            [th(i), sse(i)] = find_threshold_fast( trnX(i,:), trnY );
        end
        
        [ minSse, idx ] = min( sse );

        sse0 = sum( trnY.^2) - sum( trnY )^2 / M;

        Node.sse       = minSse;
        Node.sseImprov = sse0 - minSse;
        
        if sse0 - minSse < Opt.minSseImprov
            leafNodeCnt = leafNodeCnt + 1;
            Node.isLeaf = 1;
            Node.id     = leafNodeCnt;
            numChildren = 1;
            Node.stop   = sprintf('SSE improvement %f < minSseImprov=%f.', ...
                sse0 - minSse, Opt.minSseImprov); 
        else
            Node.isLeaf    = 0;
            Node.th        = th(idx);
            Node.inVarIdx  = idx;
        
            idxLeft        = find( trnX( idx, :) < Node.th ) ;
            idxRight       = find( trnX( idx, :) >= Node.th ) ;
        
            if numel( idxLeft ) == 0 | numel(idxRight) == 0
                % if it failed to split the data (e.g. all features are the same) 
                % then create leaf node
                leafNodeCnt = leafNodeCnt + 1;
                Node.isLeaf = 1;
                Node.id     = leafNodeCnt;
                numChildren = 1;
                Node.stop   = 'Unable to split features.';
            else
                
                [Node.LeftNode,cntL]  = regtree_createnode( trnX(:,idxLeft), trnY(idxLeft), depth+1, Opt );
                [Node.RightNode,cntR] = regtree_createnode( trnX(:,idxRight), trnY(idxRight), depth+1, Opt );
                numChildren           = cntL + cntR + 1;
            end

        end
        
    end

end


function [bestTh, bestSse] = find_threshold( trnX, trnY )
%%
% this is a naive implementation, hence very inefficient !!!

    thRange = sort( unique( trnX ), 'ascend' );
%    thRange = [thRange thRange(end)+1];   
    if length( thRange ) >= 2
        thRange = [2*thRange(1)-thRange(2) 0.5*(thRange(1:end-1)+thRange(2:end)) 2*thRange(end)-thRange(end-1)];
    else
        thRange = thRange+[-1 1];
    end
    
    bestSse = inf;
    for th = thRange;
        sseLeft  = 0;
        sseRight = 0;
        idxLeft  = find( trnX < th );
        if ~isempty( idxLeft ), sseLeft  = var( trnY(idxLeft), 1)*length(idxLeft); end
        
        idxRight = find( trnX >= th );
        if ~isempty( idxRight), sseRight = var( trnY(idxRight), 1)*length(idxRight); end
        
        if bestSse > sseLeft + sseRight
            bestSse = sseLeft + sseRight;
            bestTh  = th;
        end
    end
    
end

function [bestTh, bestSse] = find_threshold_fast( trnX, trnY )
%%
% fast implementation of "find_threshold"

    [X,idx] = sort( trnX, 'ascend' );
    Y       = trnY(idx);
    M       = numel( X );

    M1 = cumsum(Y);
    M2 = cumsum(Y.^2);
    
    x       = X(1);
    bestTh  = x - 1e-3;
    
    % bestSse = var( Y, 1 )*M;
    bestSse = M2(M) - M1(M)^2 / M;
    
    if M == 1, return; end
    
    for i = 2 : M
        if x < X(i)
            x   = X(i);
            % sse = var(Y(1:i-1),1)*(i-1) + var( Y(i:M),1)*(M-i+1);           
            sse = M2(i-1) - M1(i-1)^2 / (i-1) + (M2(M)-M2(i-1)) - (M1(M)-M1(i-1))^2/(M-i+1);
            if sse < bestSse
                bestSse = sse;
                bestTh  = 0.5*(X(i)+X(i-1));
                
                % sanity check: due to finite precision it may happen that
                % X(i-1) == 0.5*(X(i)+X(i-1)  even if X(i-1) < X(i). 
                if X(i-1) >= bestTh
                    bestTh  = X(i);
                end
            end
        end
    end
    
end




