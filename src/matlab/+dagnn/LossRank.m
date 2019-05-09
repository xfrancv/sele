classdef LossRank < dagnn.Loss
%LossMae
%

  properties
      normPower = 2;
  end

  methods
    function outputs = forward(obj, inputs, params)
        
        predictions = gather( inputs{1} );
        risk        = inputs{2}; 
        predY       = inputs{3};
        
        [~,~,nY,N]  = size( predictions );
        predictions = reshape( predictions, [nY, N]);
        
        proj = zeros(N,1);
        for i = 1 : N
            proj(i) = predictions( predY(i), i);
        end
        
        R = 0;
        for i = 1 : N
            score = 1 + proj - proj(i);
            idx   = find( score > 0 );
            R     = R + risk(i) * sum(score(idx));
        end
        R = R / N^obj.normPower;
        
        outputs{1} = R ;
      
        % Accumulate loss statistics.
        if obj.ignoreAverage, return; end;
        n = obj.numAveraged ;
        m = n + N + 1e-9 ;
        obj.average = (n * obj.average + gather(outputs{1})) / m ;
        obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        
        predictions = gather( inputs{1} );
        risk        = inputs{2}; 
        predY       = inputs{3};
        
        [~,~,nY,N]  = size( predictions );
        predictions = reshape( predictions, [nY, N]);
        
        proj = zeros(N,1);
        for i = 1 : N
            proj(i) = predictions( predY(i), i);
        end
        
        alpha = zeros( N, 1);
        for i = 1 : N
            score = 1 + proj - proj(i);
            idx   = find( score > 0 );
            alpha(idx) = alpha(idx) + risk(i);
            alpha(i)   = alpha(i) - numel(idx)*risk(i);
        end
        alpha = alpha / N^obj.normPower;
        
        der = zeros( nY, N,'single');
        for i = 1 : N
            der( predY(i), i) = alpha(i);
        end
                
        derInputs =  {reshape( der, [1 1 nY N]).*derOutputs{1},[],[]};
        derParams = {} ;
    end

    function obj = LossRank(varargin)
        obj.load(varargin) ;
    end
  end
end
