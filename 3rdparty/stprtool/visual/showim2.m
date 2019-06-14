function h = showim2(img, layout, titleStr)
% SHOWIM Displays given image(s).
%
% Synopsis:
%  h = showimg(img)
%  h = showimg(img,layout)
%  h = showimg(img,layout, titleStr)
%
%
% Input:
%  img [imH x imW x imChannels x nImages] 
%  layout [gridH x gridW] 
%
% Output:
%  h [1 x num_img] Handles of individual axes.
%

% (c) Statistical Pattern Recognition Toolbox, (C) 1999-2003,
% Written by Vojtech Franc and Vaclav Hlavac,
% <a href="http://www.cvut.cz">Czech Technical University Prague</a>,
% <a href="http://www.feld.cvut.cz">Faculty of Electrical engineering</a>,
% <a href="http://cmp.felk.cvut.cz">Center for Machine Perception</a>

% Modificatrions:
% 10-aug-2006, VF
% 15-jun-2004, VF
% 10-sep-2003, VF

    nImages = size( img, 4);

    %%  get number of rows and cols
    if nargin >= 2
        row = layout(1); 
        col = layout(2);
    else
        col=floor(sqrt(4*nImages/3));
        for i=max(1,fix(nImages/10)):fix(sqrt(nImages)),
            if nImages/i == round(nImages/i),
                col= nImages/i;
                break;
            end
        end
        row=ceil(nImages/col);
    end

    cnt = 0;
    h   = [];
    for i = 1 : nImages
        h(i) = subplot(row,col,i);
        imshow(img(:,:,:,i));
        if nargin >= 3
            title( titleStr{i});
        end
    end
end

