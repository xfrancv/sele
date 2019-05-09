function  loss = compute_loss( trueY, predY, Loss) 
% loss = compute_loss( trueY, predY, Loss) 
% 
    nY      = size(Loss,1);    
    idx     = trueY(:) + (predY(:)-1)*nY;
    loss    = Loss(idx);

end