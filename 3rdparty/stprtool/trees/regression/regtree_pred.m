function [Y,nodeId] = regtree_pred( X, Model, dbg )
%%
%   [Y,nodeId] = regtree_pred( X, Tree )
%

    %% Convert tree to decision table
    Nodes = regtree_getnodesandleafs( Model );
            
    nNodes = numel( Nodes );
    T      = zeros( nNodes, 6 );
    
    for i = 1 : nNodes
%         if i == 135
%             keyboard;
%         end
        if ~Nodes{i}.isLeaf 
            T(i,1) = i+1;
            T(i,2) = i+Nodes{i}.rightShift;
            T(i,3) = Nodes{i}.th;
            T(i,4) = Nodes{i}.inVarIdx;
            T(i,5) = nan;
            T(i,6) = nan;
        else
            T(i,1) = nan;
            T(i,2) = nan;
            T(i,3) = nan;
            T(i,4) = nan;
            T(i,5) = Nodes{i}.value;
            T(i,6) = Nodes{i}.id;
        end
    end
    

    %%
    M      = size( X, 2);
    Y      = zeros( M, 1);
    nodeId = zeros(M, 1);
    for i = 1 : M
       x = X(:,i);
       
       j = 1;
       while 1
%            if dbg == 1 & i == 5627 & j==66
%                keyboard;
%                %fprintf('i=%d,j=%d\n', i,j);
%            end
           if isnan(T(j,5))
               if isnan( T(j,4) )
                   keyboard;
               end
               if x(T(j,4)) < T(j,3)
                   j = T(j,1);
               else
                   j = T(j,2);
               end
           else
               Y(i)      = T(j,5);
               nodeId(i) = T(j,6);
               break;
           end
       end
    end


% %% Recursive implementation (nice but terribly slow in Matlab)
%
%     if isfield( Model, 'RootNode')
%         Node = Model.RootNode;
%     else
%         Node = Model;
%     end
%     
%     M      = size( X, 2);
%     Y      = zeros( M, 1);
%     nodeId = zeros(M, 1);
%     for i = 1 : M
%        x = X(:,i);
%        
%        if Node.isLeaf
%            Y(i)      = Node.value;
%            nodeId(i) = Node.id;
%        else
%            if x( Node.inVarIdx ) < Node.th
%                [Y(i),nodeId(i)] = regtree_pred( x, Node.LeftNode );
%            else
%                [Y(i),nodeId(i)] = regtree_pred( x, Node.RightNode );
%            end
%        end
%     end

end
