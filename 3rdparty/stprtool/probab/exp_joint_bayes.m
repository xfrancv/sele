
%% parameters of synthetic 2d model
Gnd.covId     = 10*eye(2,2);
Gnd.covAppear = [1 0.05; 0.05 0.1];
Gnd.meanId    = [5;5];

M = 100;  % pocet prikladu na identitu
N = 50; % pocet identit
 
%% generate training data
Trn.X = zeros(2,M*N);
Trn.Y = zeros(N*M,1);
cnt = 1;
for i = 1 : N
    mu = mvnrnd( Gnd.meanId, Gnd.covId )';    
    x  = repmat(mu,1,M) + mvnrnd( [0;0], Gnd.covAppear, M )';
    Trn.X(:,cnt:cnt+M-1) = x;
    Trn.Y(cnt:cnt+M-1)   = i;
    cnt = cnt + M;
end

%% generate testing data
nPos = 10;
nNeg = 10;
Tst.posPairs = zeros(2,2,nPos);
for i = 1 : nPos
    mu = mvnrnd( Gnd.meanId, Gnd.covId )';    
    x1 = mu + mvnrnd( [0;0], Gnd.covAppear, 1 )';
    x2 = mu + mvnrnd( [0;0], Gnd.covAppear, 1 )';
    Tst.posPairs(:,:,i) = [x1 x2];
end
Tst.negPairs = zeros(2,2,nNeg);
for i = 1 : nNeg
    mu = mvnrnd( Gnd.meanId, Gnd.covId )';    
    x1 = mu + mvnrnd( [0;0], Gnd.covAppear, 1 )';
    mu = mvnrnd( Gnd.meanId, Gnd.covId )';    
    x2 = mu + mvnrnd( [0;0], Gnd.covAppear, 1 )';
    Tst.negPairs(:,:,i) = [x1 x2];
end

%%
figure;
ppatterns( Trn.X, Trn.Y);


%% call EM algorithm
Opt.verb    = 1;
Opt.maxIter = 100;
Opt.eps     = 1e-6;
[covIntra,covExtra,State] = stg_em( Trn.X, Trn.Y, Opt );

covIntra
covExtra
%figure; plot( State.eps);


%% 
nDims = size(covIntra,1);
tmp   = inv( [ covIntra+covExtra covExtra ; covExtra covExtra+covIntra] );
H     = tmp(1:nDims,1:nDims);
G     = tmp(1:nDims,nDims+1:end);
A = inv(covIntra + covExtra) - H;

%% Test error
for i = 1 : nPos
    x1 = Tst.posPairs(:,1,i);
    x2 = Tst.posPairs(:,2,i);
    score = x1'*A*x1 + x2'*A*x2 - 2*x1'*G*x2;
    fprintf('pos pair %2d: score=%f\n', i, score);
end
for i = 1 : nNeg
    x1 = Tst.negPairs(:,1,i);
    x2 = Tst.negPairs(:,2,i);
    score = x1'*A*x1 + x2'*A*x2 - 2*x1'*G*x2;
    fprintf('neg pair %2d: score=%f\n', i, score);
end


