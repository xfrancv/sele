# Discriminative learning of prediction uncertainty

Code for ICML paper: 
V. Franc, D.Prusa: On Discriminative Learning of Prediction Uncertainty. ICML 2019.

Developed and tested in Matlab (R2016a) under Linux Ubuntu 14.04.5 LTS.


# Install

(1) Download datasets from UCI repo and libSVM webpage and convert them to MAT file. 
In Matlab issue:

    install_data

which downloads raw data and stores them to data/ folder in MAT format.
This script requires Linux tools: "uncompress, wget, bunzip2, cat". 


(2) Install MATCONVNET, LIBOCAS and STPRTOOL. This requires the Matlab MEX compiler 
to be setup properly. In Matlab run:

    install_selclassif

which installs all the packages.


# Run experiments in Matlab

(1) Set path. In Matlab issue:

    setpath_selclassif


(2) Train Logistic-regression and SVM models. In Matlab issue:

    run_all_train_classif

The script can be issued multiple times simultaneously on different computers. The function 
uses *.lock files to synchronize different instances hence the computers must have a 
shared diskdrive, namely, the folder results/.

If you have a system with Sun Grid Engine, you can issue multiple jobs automatically by 

    $ run_all_train_conf.sh


(3) Train uncertainty functions after all LR and SVM models have been trained. In Matlab run:

    run_all_train_conf

Similarly to "run_all_train_classif", this script can be issued multiple times. 
See description above.


(4) Generate EPS figures and TeX tables which appeared in the paper. In Matlab issue:

    fig_result_summary
    tab_result_summary
    tab_datasets
