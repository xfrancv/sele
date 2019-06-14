%% RSMULTI_EXAMPLE example of using reduced set method to reduce 
%   complexity (number of support vector == test time) of 
%   the multi-class SVM classifier.
%

Data = load('fiveclassproblem');
%kernelName = 'poly'; kernelArgs= [2 0 1]; C=10;
kernelName = 'rbf'; kernelArgs= 1; C=10;

newNumSV = 10; % number of support vectors of the reduced rule

%% Train multi-class SVM
Model = msvmb2( Data.X, Data.Y, C, kernelName, kernelArgs )

%% Reduce the number of SVs 
ReducedModel = rsmulti(Model,newNumSV)

%% Display results
figure; 
ppatterns(Data.X,Data.Y); 
ppatterns(Model.SV,[],'Encircle');
pclassifier(Model);

ppatterns(ReducedModel.SV,[],'BigCircles');
pclassifier(ReducedModel,[],struct('line_style','r--'));


