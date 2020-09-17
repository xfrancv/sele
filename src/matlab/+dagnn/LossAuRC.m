classdef LossAuRC < dagnn.Loss
%LossMae
%

  properties
%      normPower = 2;
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
        
        [~,idx] = sort( proj );
        auRC    = sum( cumsum( risk(idx))./[1:N]' );
                
        outputs{1} = auRC ;
      
        % Accumulate loss statistics.
        if obj.ignoreAverage, return; end;
        n = obj.numAveraged ;
        m = n + N + 1e-9 ;
        obj.average = (n * obj.average + gather(outputs{1})) / m ;
        obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        derInputs = {};
        derParams = {} ;
    end

    function obj = LossAuRC(varargin)
        obj.load(varargin) ;
    end
  end
end
