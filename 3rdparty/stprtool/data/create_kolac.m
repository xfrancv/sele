%%

dataFile = 'kolac.mat';

nY    = 5;
nTrn  = 2000;
nTst  = 5000;
C     = .2;
R     = 1;

%%
if exist( dataFile ), error('data file already exists.'); end

rng(0);

Cov   = zeros(2,2,nY);
Mean  = zeros(2,nY);
Prior = ones(nY,1)/nY;

for y = 1 : nY
    phi        = 2*pi*(y-1)/nY;
    Mean(:,y)  = R*[cos(phi); sin(phi)];
    Cov(:,:,y) = C*eye(2,2);
end

Gmm = gmm_create(Mean,Cov,Prior);

[Trn.X,Trn.Y] = gmm_samp( Gmm, nTrn);
[Tst.X,Tst.Y] = gmm_samp( Gmm, nTst);

save( dataFile, 'Trn', 'Tst', 'Gmm');

figure; hold on; 
ppatterns(Trn.X,Trn.Y);
grid on;

figure; hold on; 
ppatterns(Tst.X,Tst.Y);
grid on;
