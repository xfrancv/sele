% OCAS solver for training linear SVM classifiers from large-scale data
%
% Copyright (C) 2008,2009,2010,2012 
% Vojtech Franc, xfrancv@cmp.felk.cvut.cz
% Soeren Sonnenburg, soeren.sonnenburg@tu-berlin.de
%
% SVM solvers for training linear two-class classifiers:
%   svmocas             Accepts examples stored in dense double or sparse 
%                       double or dense single or dense int8 matrix.
%   svmocas_nnw         Allows additional constrains enforcing non-negative weights.
%   svmocas_light       Loads examples from file in SVM^light format.
%   svmocas_lbp         Examples are LBP features computed on a set of
%                       given grayscale images.
%
% SVM solver for training linear multi-class classifiers:
%   msvmocas            Accepts examples stored in dense double or 
%                       sparse double matrix.
%   msvmocas_light      Loads examples from file in SVM^light format.
%
% Auxciliary functions:
%   compute_auc             Computes area under ROC.
%   lbppyr_features         Computes LBP feature representation for given images. 
%   libocas_test            This script tests all SVM solvers in the LIBOCAS.
%   linclassif_light        Classifies examples in SVM^light file by linear rule.
%   msvmocas_light_example  Example on using multi-class SVM solver. 
%   svmocas_lbp_example     Example on training translation invariant image classifiers.
%   svmocas_parseout        Parsing out text output of SVMOCAS solver.
%