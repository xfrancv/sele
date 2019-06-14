function result=kernel_test()
% KERNEL_TEST tests functionality of KERNEL.

MAX_ALLOVED_DEVIATION = 1e-14;
result = 1;

X=rand(10,100);
Y=rand(10,50);

%% linear kernel
K = kernel(X,'linear',[]);
K_ref = X'*X;
dev = max(abs(K(:)-K_ref(:)));
if dev > MAX_ALLOVED_DEVIATION
    fprintf('KERNEL_TEST: deviation %.17f in computing K = kernel(X,linear,[])\n',dev);
    result = 0;
end

K = kernel(X,Y,'linear',[]);
K_ref = X'*Y;
dev = max(abs(K(:)-K_ref(:)));
if dev > MAX_ALLOVED_DEVIATION
    fprintf('KERNEL_TEST: deviation %.17f in computing K = kernel(X,Y,linear,[])\n',dev);
    result = 0;
end

%% rbf kernel
gamma=1;
K = kernel(X,'rbf',gamma);
M = size(X,2);
N = M;
X_sq = sum(X.^2); 
K_ref = exp(-gamma* (repmat(X_sq(:),1,N)+repmat(X_sq,M,1)-2*X'*X) );
dev = max(abs(K(:)-K_ref(:)));
if dev > MAX_ALLOVED_DEVIATION
    fprintf('KERNEL_TEST: deviation %.17f in computing K = kernel(X,rbf,%f)\n',dev,gamma);
    result = 0;
end

K = kernel(X,Y,'rbf',gamma);
M = size(X,2);
N = size(Y,2);
X_sq = sum(X.^2); 
Y_sq = sum(Y.^2); 
K_ref = exp(-gamma* (repmat(X_sq(:),1,N)+repmat(Y_sq,M,1)-2*X'*Y) );
dev = max(abs(K(:)-K_ref(:)));
if dev > MAX_ALLOVED_DEVIATION
    fprintf('KERNEL_TEST: deviation %.17f in computing K = kernel(X,Y,rbf,%f)\n',dev,gamma);
    result = 0;
end

%% poly kernel
degree = 2;
arg1 = 1;
arg0 = 1;
K = kernel(X,'poly',[degree arg0 arg1]);
K_ref = (arg1*X'*X+arg0).^degree;
dev = max(abs(K(:)-K_ref(:)));
if dev > MAX_ALLOVED_DEVIATION
    fprintf('KERNEL_TEST: deviation %.17f in computing K = kernel(X,poly,[%f %f %f])\n',dev,degree,arg0,arg1);
    result = 0;
end

degree = 2;
arg1 = 1;
arg0 = 1;
K = kernel(X,Y,'poly',[degree arg0 arg1]);
K_ref = (arg1*X'*Y+arg0).^degree;
dev = max(abs(K(:)-K_ref(:)));
if dev > MAX_ALLOVED_DEVIATION
    fprintf('KERNEL_TEST: deviation %.17f in computing K = kernel(X,Y,poly,[%f %f %f])\n',dev,degree,arg0,arg1);
    result = 0;
end

%% sigmoid kernel
arg1 = 1;
arg0 = 1;
K = kernel(X,'sigmoid',[arg0 arg1]);
K_ref = tanh( arg1*X'*X + arg0);
dev = max(abs(K(:)-K_ref(:)));
if dev > MAX_ALLOVED_DEVIATION
    fprintf('KERNEL_TEST: deviation %.17f in computing K = kernel(X,sigmoid,[%f %f])\n',dev,arg0,arg1);
    result = 0;
end

K = kernel(X,Y,'sigmoid',[arg0 arg1]);
K_ref = tanh( arg1*X'*Y + arg0);
dev = max(abs(K(:)-K_ref(:)));
if dev > MAX_ALLOVED_DEVIATION
    fprintf('KERNEL_TEST: deviation %.17f in computing K = kernel(X,Y,sigmoid,[%f %f])\n',dev,arg0,arg1);
    result = 0;
end

%% precomputed kernel
K_ref = X'*X;
K = kernel(K_ref,'precomputed',[]);
dev = max(abs(K(:)-K_ref(:)));
if dev > MAX_ALLOVED_DEVIATION
    fprintf('KERNEL_TEST: deviation %.17f in computing K = kernel(K1,precomputed,[])\n',dev);
    result = 0;
end


%%
if result == 1
    fprintf('KERNEL_TEST: Kernels (LINEAR,RBF,POLY,SIGMOID,PRECOMPUTED) passed OK\n');
end

return;
% EOF
