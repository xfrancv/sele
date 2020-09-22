classdef LossRegressionSwitch < dagnn.Loss
%LossMae
%

  properties
%      normPower = 2;
  end

  methods
    function outputs = forward(obj, inputs, params)
        
        predictions = gather( inputs{1} );
        loss        = inputs{2}; 
        predY       = inputs{3};
        
        [~,~,nY,N]  = size( predictions );
        predictions = reshape( predictions, [nY, N]);
        
        proj = zeros(N,1);
        for i = 1 : N
            proj(i) = predictions( predY(i), i);
        end
        
        outputs{1} = 0.5*sum( (loss(:)-proj(:)).^2);
        
%         loss(find(loss~=1)) = -1;        
%         outputs{1} = sum( log( 1+ exp(-proj(:).*loss(:)))) ;
      
        % Accumulate loss statistics.
        if obj.ignoreAverage, return; end;
        n = obj.numAveraged ;
        m = n + N + 1e-9 ;
        obj.average = (n * obj.average + gather(outputs{1})) / m ;
        obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        
        predictions = gather( inputs{1} );
        loss        = inputs{2}; 
        predY       = inputs{3};
        
        [~,~,nY,N]  = size( predictions );
        predictions = reshape( predictions, [nY, N]);
        
%        loss(find(loss~=1)) = -1;
        
        proj = zeros(N,1);
        for i = 1 : N
            proj(i) = predictions( predY(i), i);
        end
                
        der = zeros( nY, N,'single');
        for i = 1 : N
%            der( predY(i), i) = -loss(i)*exp(-proj(i)*loss(i))/(1+exp(-proj(i)*loss(i)));
            der( predY(i), i) = proj(i)-loss(i);
        end
                
        derInputs =  {reshape( der, [1 1 nY N]).*derOutputs{1},[],[]};
        derParams = {} ;
    end

    function obj = LossRegressionSwitch(varargin)
        obj.load(varargin) ;
    end
  end
end
