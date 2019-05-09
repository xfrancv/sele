classdef LossCustom < dagnn.Loss
%LossSmoothL1  Smooth L1 loss
%  `LossSmoothL1.forward({x, x0, w})` computes the smooth L1 distance 
%  between `x` and `x0`, weighting the elements by `w`.
%
%  Here the smooth L1 loss between two vectors is defined as:
%
%     Loss = sum_i f(x_i - x0_i) w_i.
%
%  where f is the function (following the Faster R-CNN definition):
%
%              { 0.5 * sigma^2 * delta^2,         if |delta| < 1 / sigma^2,
%   f(delta) = {
%              { |delta| - 0.5 / sigma^2,         otherwise.
%
%  In practice, `x` and `x0` can pack multiple instances as 1 x 1 x C
%  x N arrays (or simply C x N arrays).

  properties
      Loss = [];
  end

  methods
    function outputs = forward(obj, inputs, params)
        
      nY = size(inputs{1},3);
      M  = size(inputs{1},4);
      
      score    = reshape(inputs{1}, [nY M]);
      trueY    = inputs{2};
      
      post      = exp( score );
      post      = post./repmat(sum(post),nY,1);
      [~,predY] = min( obj.Loss*post );
      
%      outputs{1} = compute_cummulative_loss( trueY, predY, obj.Loss) ;
      outputs{1} = sum(compute_loss( trueY, predY, obj.Loss) );
      
      % Accumulate loss statistics.
      if obj.ignoreAverage, return; end;
      n = obj.numAveraged ;
      m = n + size( inputs{1},4) + 1e-9 ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
    % Function derivative:
    %

      derInputs = {} ;      
      derParams = {} ;
    end

    function obj = LossCustom(varargin)
      obj.load(varargin) ;
    end
  end
end
