% Example: Train linear SVM classifier by BMRMCONSTR.
%   It shows how to do efficiently model selection by recyling the 
%   CP model.
%

load('riply_dataset','Trn','Tst');

lambdaRange = 10.^[1:-1:-6];

% prepare data  
x0   = 1;     % linear rule with bias
Data = risk_svm_init( Trn.X, x0, Trn.Y );
        
%% Using constraints A*w >= b, ensure that weights are in a box [-5 5]
% A = [1 0 0; -1 0 0; 0 1 0; 0 -1 0];
% b = [-1; -1; -1; -1]*5;
A=[];
b = [];

%% Naive way to do model selection.
Opt.tolRel   = 1e-3;
%Opt.useCplex = 1;
M        = length( lambdaRange );
runTime1 = zeros(M,1);
W1       = zeros(3,M);
risk1    = zeros(M,1);
Fp1      = zeros(M,1);
for i = 1 : M
    fprintf('[Naive: lambda=%.10f]\n', lambdaRange(i)); 
    t0             = cputime;
    [W1(:,i),Stat] = bmrmconstr( Data, @risk_svm, lambdaRange(i), A, b, [], Opt );
    runTime1(i)    = cputime -t0; 
    
    risk1(i) = Stat.risk(end);
    Fp1(i)   = Stat.Fp(end);
end


%% Better way by recycling the cutting plane model
runTime2 = zeros(M,1);
W2       = zeros(3,M);
risk2    = zeros(M,1);
Fp2      = zeros(M,1);
Cpm      = [];
for i = 1 : M
    fprintf('[Recycling: lambda=%.10f]\n', lambdaRange(i)); 
    t0                 = cputime;
    [W2(:,i),Stat,Cpm] = bmrmconstr( Data, @risk_svm, lambdaRange(i), A, b, Cpm, Opt );

    runTime2(i) = cputime -t0; 
    risk2(i)    = Stat.risk(end);
    Fp2(i)      = Stat.Fp(end);
end

%% check that the solution is the same 
max(abs(risk1-risk2))
max(abs(Fp1-Fp2))
max(max(abs(W1-W2)))

%% but the runtime differs
figure;
h1=semilogx(lambdaRange,runTime1,'b'); hold on;
h2=semilogx(lambdaRange,runTime2,'r'); 
xlabel('lambda');
ylabel('cputime [s]');
legend([h1 h2],'naive (CP model from scratch)','recycling CP model');
