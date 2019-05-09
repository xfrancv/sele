function  cumLoss = compute_cumulaitve_loss( trueY, predY, Loss) 
% cumLoss = compute_cumulaitve_loss( trueY, predY, Loss) 
% 
    nY      = size(Loss,1);    
    idx     = trueY(:) + (predY(:)-1)*nY;
    cumLoss = sum( Loss(idx));

end