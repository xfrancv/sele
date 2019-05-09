function y = regtree_marginals( Tree, inVarIdx, x )
%

    N = size(x,2);
    y = zeros(1,N); 
    for i = 1 : N
        
        y(i) = get_expectation( Tree.RootNode, inVarIdx, x(:,i), 1 );        
        n    = get_expectation( Tree.RootNode, inVarIdx, x(:,i), 0 );       
        y(i) = y(i) / n;
    end
           
end

%
function E = get_expectation( Node, inVarIdx, x, flag )
    
    if Node.isLeaf 
        if flag
            E = Node.value*Node.trnExamples;
        else
            E = Node.trnExamples;
        end
    else
        idx = intersect( Node.inVarIdx, inVarIdx );
        if ~isempty( idx )
            i = find( inVarIdx == idx);
            if Node.th > x(i) 
                E = get_expectation( Node.LeftNode, inVarIdx, x, flag );
            else
                E = get_expectation( Node.RightNode, inVarIdx, x, flag  );
            end
        else
            Eleft  = get_expectation( Node.LeftNode, inVarIdx, x, flag );
            Eright = get_expectation( Node.RightNode, inVarIdx, x, flag  );
            E = [Eleft + Eright];
        end
    end

end