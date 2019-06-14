% RSPOLY2_EXAMPLE example of using RSPOLY2 to reduce complexity of kernel
%  classifier with homogeneous degree 2 polynomial kernel.
%

load('riply_dataset','trn');

kernelName = 'poly'; 
kernelArgs= [2 0 1]; 
C=10;
newNumSV = 10;

%% Train two-class SVM (currently use multi-class SVM before the two-class
% solver is implemneted)
trn.Y(find(trn.Y == -1)) = 2;

Model = msvmb2( trn.X, trn.Y, C, kernelName, kernelArgs )

Model.Alpha = Model.Alpha(1,:)-Model.Alpha(2,:);
Model.bias = Model.bias(1)-Model.bias(2);

ReducedModel = rspoly2( Model,newNumSV);

figure; 
ppatterns( trn.X, trn.Y); 
ppatterns( Model.SV,[],'Encircle');
pclassifier( Model );

ppatterns( ReducedModel.SV,[],'BigCircles');
pclassifier( ReducedModel,[],struct('line_style','r--'));


