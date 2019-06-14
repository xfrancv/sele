% This script tests functionality of SVM solvers implemneted in the LIBOCAS.
%
% The script runs SVM solvers on example data (in ./data/) and compares
% the obtained results with the solutions stored in reference files. 
% 
% This script is useful to check potential bugs e.g. introduced when 
% changing the library code.
%

% jump to root dir of libocas
cd(fileparts(which('svmocas')));

% two-class problem 
TWO_CLASS_PROBLEM = './data/riply_trn.mat';

% two-class problem in SVM^light format
TWO_CLASS_PROBLEM_SVMLIGHT = './data/riply_trn.light';

% multi-class problem
MULTI_CLASS_PROBLEM = './data/example4_train.mat';

% multi-class problem in SVM^light format
MULTI_CLASS_PROBLEM_SVMLIGHT = './data/example4_train.light';

% gender classification (male vs. female) from face images 
GENDER_IMAGE_DATABASE = './data/gender_images.mat';

% file to store/load reference solution
ReferenceFile = './data/reference_solution';

% if 1 save results to reference files else compares the results to the
% reference solutions; 
CREATE_REFERNCE_FILES = 0;

%% Solver settings
opt.C = 1;
opt.Method = 1;
opt.TolRel = 0.01;
opt.TolAbs = 0;
opt.QPBound = 0;
opt.BufSize = 500;
opt.MaxTime = inf;
opt.X0 = 1;
opt.verb = 0;

%% run all solvers for different inputs

% SVMOCAS for dense double features
fprintf('SVMOCAS: training two-class SVM classifier from dense features in double precision ...');
data = load( TWO_CLASS_PROBLEM );
[svmocasResults.W,svmocasResults.W0,svmocasResults.stat] = ...
    svmocas(data.X,opt.X0,data.y, opt.C,opt.Method,opt.TolRel,...
            opt.TolAbs,opt.QPBound,opt.BufSize,inf,opt.MaxTime,opt.verb);
fprintf('done.\n');

% SVMOCAS for dense single prec.  features
fprintf('SVMOCAS: training two-class SVM classifier from dense features in single precision ...');
data = load( TWO_CLASS_PROBLEM );
data.X = single(data.X);
[svmocasSingleResults.W,svmocasSingleResults.W0,svmocasSingleResults.stat] = ...
    svmocas(data.X,opt.X0,data.y, opt.C,opt.Method,opt.TolRel,...
            opt.TolAbs,opt.QPBound,opt.BufSize,inf,opt.MaxTime,opt.verb);
fprintf('done.\n');

% SVMOCAS for sparse double features
fprintf('SVMOCAS: training two-class SVM classifier from sparse features in double precision ...');
data = load( TWO_CLASS_PROBLEM );
data.X = sparse(data.X);
[svmocasSparseResults.W,svmocasSparseResults.W0,svmocasSparseResults.stat] = ...
    svmocas(data.X,opt.X0,data.y, opt.C,opt.Method,opt.TolRel,...
            opt.TolAbs,opt.QPBound,opt.BufSize,inf,opt.MaxTime,opt.verb);
fprintf('done.\n');

% SVMOCAS for INT8 features
fprintf('SVMOCAS: training two-class SVM classifier from dense int8 features ...');
data = load( TWO_CLASS_PROBLEM );
data.X = int8(100*data.X);
[svmocasInt8Results.W,svmocasInt8Results.W0,svmocasInt8Results.stat] = ...
    svmocas(data.X,opt.X0,data.y, opt.C,opt.Method,opt.TolRel,...
            opt.TolAbs,opt.QPBound,opt.BufSize,inf,opt.MaxTime,opt.verb);
fprintf('done.\n');

% SVMOCAS_LIGHT
fprintf('SVMOCAS_LIGHT: training two-class SVM classifier from examples stored in SVM^light file ...');
[svmocasLightResults.W,svmocasLightResults.W0,svmocasLightResults.stat] = ...
    svmocas_light(TWO_CLASS_PROBLEM_SVMLIGHT,opt.X0, opt.C,opt.Method,opt.TolRel,...
            opt.TolAbs,opt.QPBound,opt.BufSize,inf,opt.MaxTime,opt.verb);
fprintf('done.\n');

% SVMOCAS_LBP
fprintf('SVMOCAS_LBP: training two-class SVM classifier from LBP features computed on images ...');
data = load( GENDER_IMAGE_DATABASE );
HEIGHT_OF_LBP_PYRAMID = 4;
BASE_WINDOW_SIZE = [60 40];    
numMaleImages = size(data.trn_male_images,2);
numFemaleImages = size(data.trn_male_images,2);
wins = [ [1:numMaleImages numMaleImages+[1:numFemaleImages]]; ...
          repmat([20;15;0],1,numFemaleImages+numMaleImages)];
labels = [ones(1,numMaleImages) -ones(1,numFemaleImages)];    
[svmocasLBPResults.W,svmocasLBPResults.W0,svmocasLBPResults.stat] = ...
    svmocas_lbp([data.trn_male_images data.trn_female_images], data.IMAGE_SIZE,...
                 uint32(wins), BASE_WINDOW_SIZE, HEIGHT_OF_LBP_PYRAMID, opt.X0, labels, 0.001*opt.C, ...
                 opt.Method, opt.TolRel,opt.TolAbs,opt.QPBound,opt.BufSize,inf,opt.MaxTime,opt.verb);
fprintf('done.\n');

% MSVMOCAS
fprintf('MSVMOCAS: training multi-class SVM classifier from dense features in double precision ...');
data = load( MULTI_CLASS_PROBLEM );
[msvmocasResults.W,msvmocasResults.stat] = ...
    msvmocas(data.X,data.y,opt.C,opt.Method,opt.TolRel,...
            opt.TolAbs,opt.QPBound,opt.BufSize,inf,opt.MaxTime,opt.verb);
fprintf('done.\n');

% MSVMOCAS_LIGHT
fprintf('MSVMOCAS_LIGHT: training multi-class SVM classifier from examples stored in SVM^light file ...');
[msvmocasLightResults.W,msvmocasLightResults.stat] = ...
    msvmocas_light(MULTI_CLASS_PROBLEM_SVMLIGHT,opt.C,opt.Method,opt.TolRel,...
            opt.TolAbs,opt.QPBound,opt.BufSize,inf,opt.MaxTime,opt.verb);
fprintf('done.\n');

if CREATE_REFERNCE_FILES == 1,       
    %% save reference solutions to file
    fprintf('Saving reference solutions to %s\n', ReferenceFile);
    save(ReferenceFile,'svmocasResults','svmocasSparseResults','msvmocasResults',...
                       'svmocasInt8Results','svmocasLBPResults','svmocasSingleResults', ...
                       'svmocasLightResults','msvmocasLightResults');
    
else
    %% compare obtained solutions to those stored in the reference file
    ref = load(ReferenceFile);    
    
    test = [];
     
    % SVMOCAS for dense double features
    test(1).dif = sum(abs(svmocasResults.W - ref.svmocasResults.W) + ...
                      abs(svmocasResults.W0-ref.svmocasResults.W0));
    test(1).name = 'sum(|W-ref.W| + |W0-ref.W0])';
    test(2).dif = abs(svmocasResults.stat.Q_P - ref.svmocasResults.stat.Q_P);
    test(2).name = 'PrimalVal - ref.PrimalVal   ';
    test(3).dif = abs(svmocasResults.stat.Q_D - ref.svmocasResults.stat.Q_D);
    test(3).name = 'DualVal - ref.DualVal       ';
    
    fprintf('\nSVMOCAS for dense features in double precision:\n');
    for i=1:length(test)
        fprintf('   %s = %.20f ... ',test(i).name,test(i).dif);
        if test(i).dif == 0
            fprintf('SOLUTIONS EQUAL - OK\n');
        else
            fprintf('SOLUTION IS DIFFERENT!!!\n');
        end
    end
    
    % SVMOCAS for dense single precision features
    test(1).dif = sum(abs(svmocasSingleResults.W - ref.svmocasSingleResults.W) + ...
                      abs(svmocasSingleResults.W0-ref.svmocasSingleResults.W0));
    test(1).name = 'sum(|W-ref.W| + |W0-ref.W0])';
    test(2).dif = abs(svmocasSingleResults.stat.Q_P - ref.svmocasSingleResults.stat.Q_P);
    test(2).name = 'PrimalVal - ref.PrimalVal   ';
    test(3).dif = abs(svmocasSingleResults.stat.Q_D - ref.svmocasSingleResults.stat.Q_D);
    test(3).name = 'DualVal - ref.DualVal       ';
    
    fprintf('\nSVMOCAS for dense features in single precision:\n');
    for i=1:length(test)
        fprintf('   %s = %.20f ... ',test(i).name,test(i).dif);
        if test(i).dif == 0
            fprintf('SOLUTIONS EQUAL - OK\n');
        else
            fprintf('SOLUTION IS DIFFERENT!!!\n');
        end
    end
    
    % SVMOCAS for sparse double features
    test(1).dif = sum(abs(svmocasSparseResults.W - ref.svmocasSparseResults.W) + ...
                      abs(svmocasSparseResults.W0-ref.svmocasSparseResults.W0));
    test(1).name = 'sum(|W-ref.W| + |W0-ref.W0])';
    test(2).dif = abs(svmocasSparseResults.stat.Q_P - ref.svmocasSparseResults.stat.Q_P);
    test(2).name = 'PrimalVal - ref.PrimalVal   ';
    test(3).dif = abs(svmocasSparseResults.stat.Q_D - ref.svmocasSparseResults.stat.Q_D);
    test(3).name = 'DualVal - ref.DualVal       ';
    
    fprintf('\nSVMOCAS for sparse features in double precision:\n');
    for i=1:length(test)
        fprintf('   %s = %.20f ... ',test(i).name,test(i).dif);
        if test(i).dif == 0
            fprintf('SOLUTIONS EQUAL - OK\n');
        else
            fprintf('SOLUTION IS DIFFERENT!!!\n');
        end
    end
    
    % SVMOCAS for dense INT8 features
    test(1).dif = sum(abs(svmocasInt8Results.W - ref.svmocasInt8Results.W) + ...
                      abs(svmocasInt8Results.W0-ref.svmocasInt8Results.W0));
    test(1).name = 'sum(|W-ref.W| + |W0-ref.W0])';
    test(2).dif = abs(svmocasInt8Results.stat.Q_P - ref.svmocasInt8Results.stat.Q_P);
    test(2).name = 'PrimalVal - ref.PrimalVal   ';
    test(3).dif = abs(svmocasInt8Results.stat.Q_D - ref.svmocasInt8Results.stat.Q_D);
    test(3).name = 'DualVal - ref.DualVal       ';
    
    fprintf('\nSVMOCAS for dense int8 features:\n');
    for i=1:length(test)
        fprintf('   %s = %.20f ... ',test(i).name,test(i).dif);
        if test(i).dif == 0
            fprintf('SOLUTIONS EQUAL - OK\n');
        else
            fprintf('SOLUTION IS DIFFERENT!!!\n');
        end
    end
    
    % SVMOCAS_LIGHT
    test(1).dif = sum(abs(svmocasLightResults.W - ref.svmocasLightResults.W) + ...
                      abs(svmocasLightResults.W0-ref.svmocasLightResults.W0));
    test(1).name = 'sum(|W-ref.W| + |W0-ref.W0])';
    test(2).dif = abs(svmocasLightResults.stat.Q_P - ref.svmocasLightResults.stat.Q_P);
    test(2).name = 'PrimalVal - ref.PrimalVal   ';
    test(3).dif = abs(svmocasLightResults.stat.Q_D - ref.svmocasLightResults.stat.Q_D);
    test(3).name = 'DualVal - ref.DualVal       ';
    
    fprintf('\nSVMOCAS_LIGHT:\n');
    for i=1:length(test)
        fprintf('   %s = %.20f ... ',test(i).name,test(i).dif);
        if test(i).dif == 0
            fprintf('SOLUTIONS EQUAL - OK\n');
        else
            fprintf('SOLUTION IS DIFFERENT!!!\n');
        end
    end        
    
    % SVMOCAS_LBP
    test(1).dif = sum(abs(svmocasLBPResults.W - ref.svmocasLBPResults.W) + ...
                      abs(svmocasLBPResults.W0-ref.svmocasLBPResults.W0));
    test(1).name = 'sum(|W-ref.W| + |W0-ref.W0])';
    test(2).dif = abs(svmocasLBPResults.stat.Q_P - ref.svmocasLBPResults.stat.Q_P);
    test(2).name = 'PrimalVal - ref.PrimalVal   ';
    test(3).dif = abs(svmocasLBPResults.stat.Q_D - ref.svmocasLBPResults.stat.Q_D);
    test(3).name = 'DualVal - ref.DualVal       ';
    
    fprintf('\nSVMOCAS_LBP:\n');
    for i=1:length(test)
        fprintf('   %s = %.20f ... ',test(i).name,test(i).dif);
        if test(i).dif == 0
            fprintf('SOLUTIONS EQUAL - OK\n');
        else
            fprintf('SOLUTION IS DIFFERENT!!!\n');
        end
    end
    
    % MSVMOCAS
    test(1).dif = sum(sum(abs(msvmocasResults.W - ref.msvmocasResults.W)));
    test(1).name = 'sum(|W-ref.W|)           ';
    test(2).dif = abs(msvmocasResults.stat.Q_P - ref.msvmocasResults.stat.Q_P);
    test(2).name = 'PrimalVal - ref.PrimalVal';
    test(3).dif = abs(msvmocasResults.stat.Q_D - ref.msvmocasResults.stat.Q_D);
    test(3).name = 'DualVal - ref.DualVal    ';
    
    fprintf('\nMSVMOCAS:\n');
    for i=1:length(test)
        fprintf('   %s = %.20f ... ',test(i).name, test(i).dif);
        if test(i).dif == 0
            fprintf('SOLUTIONS EQUAL - OK\n');
        else
            fprintf('SOLUTION IS DIFFERENT!!!\n');
        end
    end
    
    % MSVMOCAS_LIGHT
    test(1).dif = sum(sum(abs(msvmocasLightResults.W - ref.msvmocasLightResults.W)));
    test(1).name = 'sum(|W-ref.W|)           ';
    test(2).dif = abs(msvmocasLightResults.stat.Q_P - ref.msvmocasLightResults.stat.Q_P);
    test(2).name = 'PrimalVal - ref.PrimalVal';
    test(3).dif = abs(msvmocasLightResults.stat.Q_D - ref.msvmocasLightResults.stat.Q_D);
    test(3).name = 'DualVal - ref.DualVal    ';
    
    fprintf('\nMSVMOCAS_LIGHT:\n');
    for i=1:length(test)
        fprintf('   %s = %.20f ... ',test(i).name, test(i).dif);
        if test(i).dif == 0
            fprintf('SOLUTIONS EQUAL - OK\n');
        else
            fprintf('SOLUTION IS DIFFERENT!!!\n');
        end
    end               
end

% EOF


