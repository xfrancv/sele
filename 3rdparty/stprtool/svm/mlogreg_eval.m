function posterior = mlogreg_eval(feat,W,W0)
% MLOGREG_EVAL evaluates poestior probability.
%
% Synopsis:
%  posterior = mlogreg_eval(feat,W)
%  posterior = mlogreg_eval(feat,W,W0)
%

nLabels = size(W,2)+1;

if nLabels == 1
    posterior = ones(1,size(feat,2));
    return;
end

proj = exp(W'*feat + repmat(W0(:),1,size(feat,2)));

if nLabels > 2
    sum_proj = sum(proj);
else
    sum_proj = proj;
end
posterior = [ones(1, size(feat,2)) ; proj]./repmat(1+sum_proj,nLabels,1);

return;
% EOF

