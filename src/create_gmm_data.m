function [trnX1,trnY1,trnX2,trnY2,valX2,valY2,tstX,tstY, Gmm] = create_gmm_data( dataSet )
% Generate syntheic classification data
% 
% [trnX1,trnY1,trnX2,trnY2,valX2,valY2,tstX,tstY, Gmm] = create_gmm_data( dataSet )
%
rng(0);

switch dataSet 
    case 2 % Gaussians on a circle; difficult for maxscore
        nY    = 5;
        nTrn1 = 2000;
        nTrn2 = 2000;
        nVal2 = 500;
        nTst  = 2000;
        C     = 2;
        R     = 3;

        Cov   = zeros(2,2,nY);
        Mean  = zeros(2,nY);
        Prior = ones(nY,1)/nY;

        for y = 1 : nY
            phi        = 2*pi*(y-1)/nY;
            Mean(:,y)  = R*[cos(phi); sin(phi)];
            v1 = C*[cos(phi+0.5*pi); sin(phi+0.5*pi)];
            v2 = C*0.25*[cos(phi); sin(phi)];
            Cov(:,:,y) = [v1 v2]*[v1 v2]';
        end

        Gmm = gmm_create(Mean,Cov,Prior);
        [trnX1,trnY1] = gmm_samp( Gmm, nTrn1);
        [trnX2,trnY2] = gmm_samp( Gmm, nTrn2);
        [valX2,valY2] = gmm_samp( Gmm, nVal2);
        [tstX,tstY] = gmm_samp( Gmm, nTst);
        
    case 1 % Gaussians on a circle; easy for max-score
        nY    = 5;
        nTrn1 = 2000;
        nTrn2 = 2000;
        nVal2 = 500;
        nTst  = 2000;
        C     = 2;
        R     = 3;

        Cov   = zeros(2,2,nY);
        Mean  = zeros(2,nY);
        Prior = ones(nY,1)/nY;

        for y = 1 : nY
            phi        = 2*pi*(y-1)/nY;
            Mean(:,y)  = R*[cos(phi); sin(phi)];
            Cov(:,:,y) = C*eye(2,2);
        end

        Gmm = gmm_create(Mean,Cov,Prior);
        [trnX1,trnY1] = gmm_samp( Gmm, nTrn1);
        [trnX2,trnY2] = gmm_samp( Gmm, nTrn2);
        [valX2,valY2] = gmm_samp( Gmm, nVal2);
        [tstX,tstY] = gmm_samp( Gmm, nTst);
      
    case 3 % two overlaping Gaussians
        nTrn1 = 2000;
        nTrn2 = 2000;
        nVal2 = 500;
        nTst  = 2000;
        Mean  = [[-1;0] [1;0]];
        Cov   = cat(3, [3 0;0 0.5], [0.5 0; 0 3]);
        prior = [1 1]/3;
        Gmm   = gmm_create(Mean,Cov,prior);

        [trnX1,trnY1] = gmm_samp( Gmm, nTrn1);
        [trnX2,trnY2] = gmm_samp( Gmm, nTrn2);
        [valX2,valY2] = gmm_samp( Gmm, nVal2);
        [tstX,tstY] = gmm_samp( Gmm, nTst);
        
end
