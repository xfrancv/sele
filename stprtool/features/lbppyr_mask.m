function featMask = lbppyr_mask(M,heightOfPyramid);
% LBPPYR_MASK
% Synopsis:
%  featMask = lbppyr_mask(M,heightOfPyramid)
%
% heightOfPyramid = 4;
% maskImg = imread('test_mask.png');
% M = maskImg > 0;
%
% featMask = lbppyr_mask(M,heightOfPyramid);
% 
% I = rand(size(M))*255; 
%
% F1 = lbppyr(uint8(I),heightOfPyramid);
%
% F2 = lbppyr(uint8(I.*M),heightOfPyramid);
% sum(F1(find(featMask))~=F2(find(featMask)) )
%
%

[imH,imW] = size(M);

F0 = lbppyr(uint8(M(:)), heightOfPyramid);
nDims = length(F0);

featMask = logical(zeros(nDims,1));
cnt = 1;
for l=1:heightOfPyramid
    lbp_win_size = 3*2^(l-1);

    for x=1:2^(l-1):imW-lbp_win_size+1
        for y=1:2^(l-1):imH-lbp_win_size+1
            win = M(y:y+lbp_win_size-1,x:x+lbp_win_size-1);
            
            featMask(cnt:cnt+255) = all(win(:));
            cnt = cnt + 256;
        end
    end
end

return;
