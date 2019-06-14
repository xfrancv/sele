%
clc;
C=100;

X=int8([0 1 0 1;0 0 1 1;zeros(6,4);0 1 0 1;0 0 1 1;zeros(6,4)]); 
y = [+1 +1 -1 -1];
[W,W0,stat] = svmocas(X,1,y,C);

%%%

Xbool = uint8([0 1 2 3; 0 1 2 3]);
y = [+1 +1 -1 -1];
[Wbool,W0bool,stat] = svmocas_bool(Xbool,1,y,C);

W-Wbool
W0-W0bool
