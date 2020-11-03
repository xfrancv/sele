classdef LossRankLog < dagnn.Loss
%LossMae
%

  properties
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
        
 %       R = 0;
%        nconst = N^2*log(2);
        R = 0;
        for i = 1 : N
            score = proj - proj(i);
         %   R        = R + risk(i) * sum( logsumexp( [score(:)' ; zeros(1,N)])) / nconst;
            
            R = R + risk(i) * sum( log(1+exp(score) ) );
            
%             idx   = find( score > 0 );
%             R     = R + risk(i) * sum(score(idx));
        end        
        outputs{1} = R/N ;
      
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
            score    = proj - proj(i);
            expScore = exp( score);
            A        = risk(i)*(expScore./(1+expScore));
            alpha    = alpha + A(:);
            alpha(i) = alpha(i) - sum(A);
            
%             idx   = find( score > 0 );
%             alpha(idx) = alpha(idx) + risk(i);
%             alpha(i)   = alpha(i) - numel(idx)*risk(i);
        end
        alpha = alpha / N;
        
        der = zeros( nY, N,'single');
        for i = 1 : N
            der( predY(i), i) = alpha(i);
        end
                
        derInputs =  {reshape( der, [1 1 nY N]).*derOutputs{1},[],[]};
        derParams = {} ;
    end

    function obj = LossRiskLog(varargin)
    end
  end
end
