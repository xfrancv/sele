/*=================================================================================
 * msvmocas_light_mex.c: OCAS solver for training multi-class linear SVM classifiers
 *                       loading examples from SVM^light file.
 * 
 * Synopsis:
 *   [W,stat] = msvmocas_light(dataFile,C,Method,TolRel,TolAbs,QPBound,BufSize,nExamples,MaxTime,verb)
 * 
 * Input:
 *  dataFile [string] path to file with training examples in SVM^light format
 *  X [nDim x nExamples] training feature inputs (sparse or dense matrix of doubles).
 *  y [nExamples x 1] labels; intgers 1,2,...nY
 *  C [1x1] regularization constant
 *  Method [1x1] 0..cutting plane; 1..OCAS  (default 1)
 *  TolRel [1x1] halts if Q_P-Q_D <= abs(Q_P)*TolRel  (default 0.01)
 *  TolAbs [1x1] halts if Q_P-Q_D <= TolAbs  (default 0)
 *  QPValue [1x1] halts if Q_P <= QPBpound  (default 0)
 *  BufSize [1x1] Initial size of active constrains buffer (default 2000)
 *  nExamples [1x1] Number of training examplesused for training; must be >0 and <= size(X,2).
 *     If nExamples = inf then nExamples is set to size(X,2).
 *  MaxTime [1x1] halts if time used by solver (data loading time is not counted) exceeds
 *    MaxTime given in seconds. Use MaxTime=inf (default) to switch off this stopping condition. 
 *  verb [1x1] if non-zero then prints some info; (default 1)
 *
 * Output:
 *  W [nDim x nY] Paramater vectors of decision rule; [dummy,ypred] = max(W'*x)
 *  stat [struct] Optimizer statistics (field names are self-explaining).
 * 
 * Copyright (C) 2008 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public 
 * License as published by the Free Software Foundation; 
 *======================================================================================*/ 

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <mex.h>

/*#define LIBOCAS_MATLAB*/

#include "libocas.h"
#include "ocas_helper.h"
#include "features_double.h"

#define DEFAULT_METHOD 1
#define DEFAULT_TOLREL 0.01
#define DEFAULT_TOLABS 0.0
#define DEFAULT_QPVALUE 0.0
#define DEFAULT_BUFSIZE 2000
#define DEFAULT_MAXTIME mxGetInf()
#define DEFAULT_VERB 1



/*======================================================================
  Main code plus interface to Matlab.
========================================================================*/

void mexFunction( int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[] )
{
  double C, TolRel, TolAbs, MaxTime, trn_err, QPBound;
  double *ptr;
  uint32_t i, j, BufSize;
  uint16_t Method;
  int verb;
  ocas_return_value_T ocas;

  /* timing variables */
  double init_time;
  double total_time;

  total_time = get_time();
  init_time = total_time;

  if(nrhs < 2 || nrhs > 10)
    mexErrMsgTxt("Improper number of input arguments.\n"
		 "\n"
		 "OCAS solver for training multi-class linear SVM classifiers.\n"
		 "\n"
                 "Synopsis:\n"
		 "  [W,stat] = msvmocas(dataFile,C,Method,TolRel,TolAbs,QPBound,BufSize,nExamples,MaxTime,verb)\n"
		 "\n"
		 "Input:\n"
		 "  dataFile [string] path to file with training examples in SVM^light format\n"
		 "  y [nExamples x 1] labels must be integers 1,2,...nY\n"
		 "  C [1x1] regularization constant\n"
		 "  Method [1x1] 0..cutting plane; 1..OCAS  (default 1)\n"
		 "  TolRel [1x1] halts if Q_P-Q_D <= abs(Q_P)*TolRel  (default 0.01)\n"
		 "  TolAbs [1x1] halts if Q_P-Q_D <= TolAbs  (default 0)\n"
		 "  QPValue [1x1] halts if Q_P <= QPBpound  (default 0)\n"
		 "  BufSize [1x1] Initial size of active constrains buffer (default 2000)\n"
		 "  nExamples [1x1] Number of training examples used for training; must be >0 and <= size(X,2).\n"
		 "    If nExamples = inf then nExamples is set to size(X,2).\n"
		 "  MaxTime [1x1] halts if time used by solver (data loading time is not counted) exceeds\n"
		 "    MaxTime given in seconds. Use MaxTime=inf (default) to switch off this stopping condition.\n"
		 "  verb [1x1] if non-zero then prints some info; (default 1)\n"
		 "\n"
		 "Output:\n"
		 "  W [nDim x nY] Paramater vectors of decision rule; [dummy,ypred] = max(W'*x)\n"
		 "  stat [struct] Optimizer statistics (field names are self-explaining).\n");

  char *fname;
  int fname_len;

  if(!mxIsChar(prhs[0]))
    mexErrMsgTxt("First input argument must be of type string.");

  fname_len = mxGetNumberOfElements(prhs[0]) + 1;   
  fname = mxCalloc(fname_len, sizeof(char));    

  if (mxGetString(prhs[0], fname, fname_len) != 0)     
    mexErrMsgTxt("Could not convert first input argument to string.");

  if(nrhs >= 10)
    verb = (int)mxGetScalar(prhs[9]);
  else
    verb = DEFAULT_VERB;

  /* load data */
  if( load_svmlight_file(fname,verb) == -1 || data_X == NULL || data_y == NULL)
    mexErrMsgTxt("Cannot load input file.");

  C = (double)mxGetScalar(prhs[1]);

  if(nrhs >= 3)
    Method = (uint32_t)mxGetScalar(prhs[2]);
  else
    Method = DEFAULT_METHOD;

  if(nrhs >= 4)
    TolRel = (double)mxGetScalar(prhs[3]);
  else
    TolRel = DEFAULT_TOLREL;

  if(nrhs >= 5)    
    TolAbs = (double)mxGetScalar(prhs[4]);
  else
    TolAbs = DEFAULT_TOLABS;

  if(nrhs >= 6)
    QPBound = (double)mxGetScalar(prhs[5]);
  else
    QPBound = DEFAULT_QPVALUE;
    
  if(nrhs >= 7)
    BufSize = (uint32_t)mxGetScalar(prhs[6]);
  else
    BufSize = DEFAULT_BUFSIZE;

  if(nrhs >= 8 && mxIsInf(mxGetScalar(prhs[7])) == false)
    nData = (uint32_t)mxGetScalar(prhs[7]);
  else
    nData = mxGetN(data_X);

  if(nData < 1 || nData > mxGetN(data_X)) 
    mexErrMsgTxt("Improper value of argument nData.");

  if(nrhs >= 9)
    MaxTime = (double)mxGetScalar(prhs[8]);
  else
    MaxTime = DEFAULT_MAXTIME;


/*  nDim = mxGetM(prhs[0]);*/
  nDim = mxGetM(data_X);
  for(i=0, nY = 0; i < nData; i++) 
  {
      nY = LIBOCAS_MAX(nY, (uint32_t)data_y[i]);
  }

  /*----------------------------------------------------------------
    Print setting
  -------------------------------------------------------------------*/
  if(verb)
  {
    mexPrintf("Input data statistics:\n"
              "   # of examples  : %d\n"
              "   # of classes   : %d\n"
              "   dimensionality : %d\n",
              nData, nY, nDim);
    
    if( mxIsSparse(data_X)== true ) 
      mexPrintf("   density        : %.2f%%\n",
                100.0*(double)mxGetNzmax(data_X)/((double)nDim*(double)(mxGetN(data_X))));
    else
      mexPrintf("    density       : 100%% (full)\n");

    mexPrintf("Setting:\n"
         "   C              : %f\n"
         "   # of examples  : %d\n"
         "   solver         : %d\n"
         "   cache size     : %d\n"
         "   TolAbs         : %f\n"
         "   TolRel         : %f\n"
         "   QPValue        : %f\n"
         "   MaxTime        : %f [s]\n",
         C, nData, Method,BufSize,TolAbs,TolRel, QPBound, MaxTime);
  }
  
  /* learned weight vector */
  plhs[0] = (mxArray*)mxCreateDoubleMatrix(nDim,nY,mxREAL);
  W = (double*)mxGetPr(plhs[0]);
  if(W == NULL) mexErrMsgTxt("Not enough memory for vector W.");

  oldW = (double*)mxCalloc(nY*nDim,sizeof(double));
  if(oldW == NULL) mexErrMsgTxt("Not enough memory for vector oldW.");

  /* allocate buffer for computing cutting plane */
  new_a = (double*)mxCalloc(nY*nDim,sizeof(double));
  if(new_a == NULL) 
    mexErrMsgTxt("Not enough memory for auxciliary cutting plane buffer new_a.");  

  /* select function to print progress info */
  void (*print_function)(ocas_return_value_T);
  if(verb) 
  {
    mexPrintf("Starting optimization:\n");
    print_function = &ocas_print;
  }
  else 
  {
    print_function = &ocas_print_null;
  }

  if( mxIsSparse(data_X)== true ) 
  {
    /* init cutting plane buffer */
    sparse_A.nz_dims = mxCalloc(BufSize,sizeof(uint32_t));
    sparse_A.index = mxCalloc(BufSize,sizeof(sparse_A.index[0]));
    sparse_A.value = mxCalloc(BufSize,sizeof(sparse_A.value[0]));
    if(sparse_A.nz_dims == NULL || sparse_A.index == NULL || sparse_A.value == NULL) 
      mexErrMsgTxt("Not enough memory for cutting plane buffer sparse_A.");  

    init_time=get_time()-init_time;

    ocas = msvm_ocas_solver( C, data_y, nY, nData, TolRel, TolAbs, QPBound, MaxTime,BufSize, Method,
                             &msvm_sparse_compute_W, &msvm_update_W, &msvm_sparse_add_new_cut,
                             &msvm_sparse_compute_output, &qsort_data, print_function, 0);
  }
  else
  {
    /* init cutting plane buffer */
    full_A = mxCalloc(BufSize*nDim*nY,sizeof(double));
    if( full_A == NULL )
      mexErrMsgTxt("Not enough memory for cutting plane buffer full_A.");    

    init_time=get_time()-init_time;

    ocas = msvm_ocas_solver( C, data_y, nY, nData, TolRel, TolAbs, QPBound, MaxTime,BufSize, Method,
                             &msvm_full_compute_W, &msvm_update_W, &msvm_full_add_new_cut,
                             &msvm_full_compute_output, &qsort_data, print_function, 0); 

  }

  total_time=get_time()-total_time;
 
  if(verb)
  {
    mexPrintf("Stopping condition: ");
    switch( ocas.exitflag )
    {
       case 1: mexPrintf("1-Q_D/Q_P <= TolRel(=%f) satisfied.\n", TolRel); break;
       case 2: mexPrintf("Q_P-Q_D <= TolAbs(=%f) satisfied.\n", TolAbs); break;
       case 3: mexPrintf("Q_P <= QPBound(=%f) satisfied.\n", QPBound); break;
       case 4: mexPrintf("Optimization time (=%f) >= MaxTime(=%f).\n", ocas.ocas_time, MaxTime); break;
       case -1: mexPrintf("Has not converged!\n" ); break;
       case -2: mexPrintf("Not enough memory for the solver.\n" ); break;
    }
  
    mexPrintf("Timing statistics:\n"
			"   init_time      : %f[s]\n"
			"   qp_solver_time : %f[s]\n"
			"   sort_time      : %f[s]\n"
			"   output_time    : %f[s]\n"
			"   add_time       : %f[s]\n"
			"   w_time         : %f[s]\n"
			"   print_time     : %f[s]\n"
			"   ocas_time      : %f[s]\n"
			"   total_time     : %f[s]\n",
			init_time, ocas.qp_solver_time, ocas.sort_time, ocas.output_time, 
            ocas.add_time, ocas.w_time, ocas.print_time, ocas.ocas_time, total_time);

    mexPrintf("Training error: %.4f%%\n", 100*(double)ocas.trn_err/(double)nData);
  }

  const char *field_names[] = {"nTrnErrors","Q_P","Q_D","nIter","nCutPlanes","exitflag",
                               "init_time","output_time","sort_time","qp_solver_time","add_time",
                               "w_time","ocas_time","total_time"}; 
  mwSize dims[2] = {1,1};  

  plhs[1] = mxCreateStructArray(2, dims, (sizeof(field_names)/sizeof(*field_names)), field_names);
  
  mxSetField(plhs[1],0,"nIter",mxCreateDoubleScalar((double)ocas.nIter));
  mxSetField(plhs[1],0,"nCutPlanes",mxCreateDoubleScalar((double)ocas.nCutPlanes));
  mxSetField(plhs[1],0,"nTrnErrors",mxCreateDoubleScalar(ocas.trn_err)); 
  mxSetField(plhs[1],0,"Q_P",mxCreateDoubleScalar(ocas.Q_P)); 
  mxSetField(plhs[1],0,"Q_D",mxCreateDoubleScalar(ocas.Q_D)); 
  mxSetField(plhs[1],0,"init_time",mxCreateDoubleScalar(init_time)); 
  mxSetField(plhs[1],0,"output_time",mxCreateDoubleScalar(ocas.output_time)); 
  mxSetField(plhs[1],0,"sort_time",mxCreateDoubleScalar(ocas.sort_time)); 
  mxSetField(plhs[1],0,"qp_solver_time",mxCreateDoubleScalar(ocas.qp_solver_time)); 
  mxSetField(plhs[1],0,"add_time",mxCreateDoubleScalar(ocas.add_time)); 
  mxSetField(plhs[1],0,"w_time",mxCreateDoubleScalar(ocas.w_time)); 
  mxSetField(plhs[1],0,"ocas_time",mxCreateDoubleScalar(ocas.ocas_time)); 
  mxSetField(plhs[1],0,"total_time",mxCreateDoubleScalar(total_time)); 
  mxSetField(plhs[1],0,"exitflag",mxCreateDoubleScalar((double)ocas.exitflag)); 

  return;
}

