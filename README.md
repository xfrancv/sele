# SELE: discriminative learning of selective classifiers

This repository contains Matlab implementation of a method for learning selective
classifiers which was published in

V. Franc, D.Prusa: On Discriminative Learning of Prediction Uncertainty. ICML 2019.

It was developed and tested in Matlab (R2016a) under Linux Ubuntu 18.04.2 LTS.


# Install

You need to download MATCONVNET and compile all MEX files. To this end go to MATLAB
and run 

`selclassif_install`


If you intend to replicate the experiments from ICML paper you will need to
download the UCI and LIBSVM datasets and convert then to MAT files. This can be done
by running 

`selclassif_install_data`


# Demo

There is a demo which shows how to a train selective classifier. The predictor
is the multi-class SVM classifier. Then, there are 4 different methods how to
construct the selection function. The theory behind is
outlined [here](http://cmp.felk.cvut.cz/~xfrancv/pages/sele.html). In Matlab run

`example_svm`

which will train all the selective classifiera, it will compare their performance in terms of
Risk-Coverage curve and it will visualize them in 2D. The resulting figures are stored
to the folder results/.


# ICML paper Experiments

The codes for ICML paper are all stored in the folder icml2019/. To replicate the results, do
the following 3 steps:

(1) Train Logistic-Regression and SVM models. In Matlab issue:

`run_all_train_classif`

This script can be issued multiple times simultaneously on different computers. The function 
uses *.lock files to synchronize different instances hence the computers must have a 
shared diskdrive, namely, the folder results/.

If you have a system with Sun Grid Engine, you can issue multiple jobs automatically by 

$ run_all_train_classif.sh

(2) Train uncertainty functions after all LR and SVM models have been trained. In Matlab run:

`run_all_train_conf`

Similarly to "run_all_train_classif", this script can be issued multiple times. See
the description above. The corresponding batch script is "run_all_train_classif.sh".


(3) Generate EPS figures and TeX tables which appeared in the paper. In Matlab issue:

```
fig_result_summary
tab_result_summary
tab_datasets
```


