/*======================================================================================
 * svmocas_mex.c: Matlab MEX interface to OCAS solver for training two-class 
 *                linear SVM classifier
 *
 * Synopsis:
 *  [W,W0,stat] = svmocas(X,X0,y,C,Method,TolRel,TolAbs,QPBound,BufSize,
 *                        nData,MaxTime,verb)
 *
 * See svmocas.m for more help.
 *
 * Copyright (C) 2008, 2009 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 *                          Soeren Sonnenburg, soeren.sonnenburg@first.fraunhofer.de
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public 
 * License as published by the Free Software Foundation; 
 *=====================================================================================*/ 

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <mex.h>

#include "libocas.h"
#include "ocas_helper.h"
#include "features_int8.h"
#include "features_double.h"
#include "features_single.h"

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
  double C, TolRel, TolAbs, QPBound, trn_err, MaxTime;
  double *vec_C;   
  uint32_t num_of_Cs;
  uint32_t i, j, BufSize;
  uint16_t Method;
  int verb;
  double *ptr_double;
  float *ptr_single;
  int8_t *ptr_int8;
  ocas_return_value_T ocas;

  /* timing variables */
  double init_time;
  double total_time;

  total_time = get_time();
  init_time = total_time;

  if(nrhs < 4 || nrhs > 12)
    mexErrMsgTxt("Improper number of input arguments.\n\n"
                 "SVMOCAS solver for training two-class linear SVM classifiers\n\n"
                 "Synopsis:\n"
                 "  [W,W0,stat] = svmocas(X,X0,y,C,Method,TolRel,TolAbs,QPBound,"
                 "BufSize,nExamples,MaxTime) \n\n"
                 "Input:  \n"
                 "  X [nDim x nExamples] training inputs (dense double or sparse double or dense single or dense int8)\n"
                 "  X0 [1 x 1 (double)] constant feature added to all examples\n"
                 "  y [nExamples x 1 (double)] labels of the examples (+1/-1)\n"
                 "  C [1x1]  or [nExamples x 1] regularization constant(s) \n"
                 "  Method [1x1 (double)] 0 for BMRM; 1 for OCAS \n"
                 "  TolRel [1x1 (double)]\n"
                 "  TolAbs [1x1 (double)]\n"
                 "  QPBound [1x1 (double)]\n"
                 "  BufSize [1x1 (double)]\n"
                 "  nExamples [1x1 (double) number of examples to use; "
                 "(inf means use all examples)\n"
                 "  MaxTime [1x1 (double)]\n"
                 "  verb [1x1 (bouble)]\n\n"
                 "Output:\n"
                 "  W [nDim x 1] Parameter vector\n"
                 "  W0 [1x1] Bias term\n"
                 "  stat [struct] \n");


  if(nrhs >= 12)
    verb = (int)mxGetScalar(prhs[11]);
  else
    verb = DEFAULT_VERB;

  /* 1st input argument: training feature vectors */
  data_X = (mxArray*)prhs[0];
  if( (mxGetNumberOfDimensions(data_X) != 2) ||
      !( ( mxIsDouble(data_X) && mxIsSparse(data_X)  ) ||
         ( mxIsDouble(data_X) && !mxIsSparse(data_X) ) ||
         ( mxIsSingle(data_X) && !mxIsSparse(data_X) ) ||
         ( mxIsInt8(data_X)   && !mxIsSparse(data_X) ) ))
  {
    mexErrMsgTxt("The first input argument must be two dimensional matrix of the following type:\n"
		 "dense double or sparse double or dense single or dense int8 matrix.\n");
  }
  
  /* 2nd  input argument: constant coordinate added to feature vectors */
  X0 = (double)mxGetScalar(prhs[1]);

  /*3rd input argument: vector of labels */
  if( !mxIsDouble(prhs[2]) || mxIsSparse(prhs[2]) )
    mexErrMsgTxt("The third input argument must be dense vector of doubles.");
  data_y = (double*)mxGetPr(prhs[2]);

  if(LIBOCAS_MAX(mxGetM(prhs[2]),mxGetN(prhs[2])) != mxGetN(prhs[0]))
    mexErrMsgTxt("Length of vector y (3rd input argument) must equl to the number of columns of matrix X (1st input argument).");

  nDim = mxGetM(prhs[0]);

  if(verb)
  {
    mexPrintf("Input data statistics:\n"
              "   # of examples  : %d\n"
              "   dimensionality : %d\n",
              mxGetN(data_X), nDim);
    
    if( mxIsSparse(data_X)== true ) 
      mexPrintf("   sparse features (density=%.2f%%) ",
               100.0*(double)mxGetNzmax(data_X)/((double)nDim*(double)(mxGetN(data_X))));
    else
      mexPrintf("   dense features ");
    if( mxIsDouble(data_X) )
      mexPrintf("in double precision\n");
    if( mxIsSingle(data_X) )
      mexPrintf("in single precision\n");
    if( mxIsInt8(data_X) )
      mexPrintf("represented as int8\n");
  }

  num_of_Cs = LIBOCAS_MAX(mxGetN(prhs[3]),mxGetM(prhs[3]));
  if(num_of_Cs == 1)
  {
     C = (double)mxGetScalar(prhs[3]);
  }
  else
  {
     vec_C = (double*)mxGetPr(prhs[3]);
  }

  if(nrhs >= 5)
    Method = (uint32_t)mxGetScalar(prhs[4]);
  else
    Method = DEFAULT_METHOD;

  if(nrhs >= 6)
    TolRel = (double)mxGetScalar(prhs[5]);
  else
    TolRel = DEFAULT_TOLREL;

  if(nrhs >= 7)    
    TolAbs = (double)mxGetScalar(prhs[6]);
  else
    TolAbs = DEFAULT_TOLABS;

  if(nrhs >= 8)
    QPBound = (double)mxGetScalar(prhs[7]);
  else
    QPBound = DEFAULT_QPVALUE;

  if(nrhs >= 9)
    BufSize = (uint32_t)mxGetScalar(prhs[8]);
  else
    BufSize = DEFAULT_BUFSIZE;

  if(nrhs >= 10 && mxIsInf(mxGetScalar(prhs[9])) == false)
    nData = (uint32_t)mxGetScalar(prhs[9]);
  else
    nData = mxGetN(data_X);
      
  if(nData < 1 || nData > mxGetN(prhs[0])) 
    mexErrMsgTxt("Improper value of argument nData.");

  if(num_of_Cs > 1 && num_of_Cs < nData)
    mexErrMsgTxt("Length of the vector C less than the number of examples.");

  if(nrhs >= 11)
    MaxTime = (double)mxGetScalar(prhs[10]);
  else
    MaxTime = DEFAULT_MAXTIME;


  /*----------------------------------------------------------------
    Print setting
  -------------------------------------------------------------------*/
  if(verb)
  {
    mexPrintf("Setting:\n");

    if( num_of_Cs == 1)
      mexPrintf("   C              : %f\n", C);
    else
      mexPrintf("   C              : different for each example\n");

    mexPrintf("   bias           : %.0f\n"
              "   # of examples  : %d\n"
              "   solver         : %d\n"
              "   cache size     : %d\n"
              "   TolAbs         : %f\n"
              "   TolRel         : %f\n"
              "   QPValue        : %f\n"
              "   MaxTime        : %f [s]\n"
              "   verb           : %d\n",
              X0, nData, Method,BufSize,TolAbs,TolRel, QPBound, MaxTime, verb);
  }
  
  /* learned weight vector */
  plhs[0] = (mxArray*)mxCreateDoubleMatrix(nDim,1,mxREAL);
  W = (double*)mxGetPr(plhs[0]);
  if(W == NULL) mexErrMsgTxt("Not enough memory for vector W.");

  oldW = (double*)mxCalloc(nDim,sizeof(double));
  if(oldW == NULL) mexErrMsgTxt("Not enough memory for vector oldW.");

  W0 = 0;
  oldW0 = 0;

  A0 = mxCalloc(BufSize,sizeof(A0[0]));
  if(A0 == NULL) mexErrMsgTxt("Not enough memory for vector A0.");

  /* allocate buffer for computing cutting plane */
  new_a = (double*)mxCalloc(nDim,sizeof(double));
  if(new_a == NULL) 
    mexErrMsgTxt("Not enough memory for auxciliary cutting plane buffer new_a.");  

  if(num_of_Cs > 1)
  {
    for(i=0; i < nData; i++) 
      data_y[i] = data_y[i]*vec_C[i];
  }

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

    /* for i=1:nData, X(:,i) = X(:,i)*y(i); end*/
    for(i=0; i < nData; i++) 
        mul_sparse_col(data_y[i], data_X, i);           
  

    /* init cutting plane buffer */
    sparse_A.nz_dims = mxCalloc(BufSize,sizeof(uint32_t));
    sparse_A.index = mxCalloc(BufSize,sizeof(sparse_A.index[0]));
    sparse_A.value = mxCalloc(BufSize,sizeof(sparse_A.value[0]));
    if(sparse_A.nz_dims == NULL || sparse_A.index == NULL || sparse_A.value == NULL) 
      mexErrMsgTxt("Not enough memory for cutting plane buffer sparse_A.");  

    init_time=get_time()-init_time;


    if(num_of_Cs == 1)
    {
      ocas = svm_ocas_solver( C, nData, TolRel, TolAbs, QPBound, MaxTime,BufSize, Method, 
                              &sparse_compute_W, &update_W, &sparse_add_new_cut, 
                              &sparse_compute_output, &qsort_data, 
                              print_function, 0);
    }  
    else
    {
      ocas = svm_ocas_solver_difC( vec_C, nData, TolRel, TolAbs, QPBound, 
                                   MaxTime,BufSize, Method, 
                                   &sparse_compute_W, &update_W, 
                                   &sparse_add_new_cut, &sparse_compute_output, 
                                   &qsort_data, print_function, 0);
    }  

  }
  else
  {

    int (*add_new_cut)(double*, uint32_t*, uint32_t, uint32_t, void*);
    int (*compute_output)( double*, void* );    
    
    /* features in double precision */
    if( mxIsDouble(data_X) )
    {
      ptr_double = mxGetPr(data_X);
      for(i=0; i < nData; i++) {
	for(j=0; j < nDim; j++ ) {
	  ptr_double[LIBOCAS_INDEX(j,i,nDim)] = ptr_double[LIBOCAS_INDEX(j,i,nDim)]*data_y[i];
	}
      }
      
      add_new_cut = &full_add_new_cut;
      compute_output = &full_compute_output;
    }

    /* features in single precision */
    if( mxIsSingle(data_X) )
    {
      ptr_single = (float*)mxGetPr(data_X);
      for(i=0; i < nData; i++) {
	for(j=0; j < nDim; j++ ) {
	  ptr_single[LIBOCAS_INDEX(j,i,nDim)] = ptr_single[LIBOCAS_INDEX(j,i,nDim)]*data_y[i];
	}
      }
      
      add_new_cut = &full_single_add_new_cut;
      compute_output = &full_single_compute_output;
    }
    
    /* features in int8  */
    if( mxIsInt8(data_X) )
    {
      ptr_int8 = (int8_t*)mxGetPr(data_X);
      for(i=0; i < nData; i++) {
	for(j=0; j < nDim; j++ ) {
	  ptr_int8[LIBOCAS_INDEX(j,i,nDim)] = ptr_int8[LIBOCAS_INDEX(j,i,nDim)]/(int8_t)data_y[i];
	}
      }
      
      add_new_cut = &full_int8_add_new_cut;
      compute_output = &full_int8_compute_output;
    }    

    /* init cutting plane buffer */
    full_A = mxCalloc(BufSize*nDim,sizeof(double));
    if( full_A == NULL )
      mexErrMsgTxt("Not enough memory for cutting plane buffer full_A.");    

    init_time=get_time()-init_time;
    
    if(num_of_Cs == 1)
    {
      ocas = svm_ocas_solver( C, nData, TolRel, TolAbs, QPBound, MaxTime,BufSize, Method, 
                              &full_compute_W, &update_W, add_new_cut, 
                              compute_output, &qsort_data, print_function, 0);
    }
    else
    {
      ocas = svm_ocas_solver_difC( vec_C, nData, TolRel, TolAbs, QPBound, MaxTime,
                                   BufSize, Method, 
                                   &full_compute_W, &update_W, add_new_cut, 
                                   compute_output, &qsort_data, print_function, 0);
    }
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
       case 4: mexPrintf("Optimization time (=%f) >= MaxTime(=%f).\n", 
                          ocas.ocas_time, MaxTime); break;
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

  /* multiply data by labels as it was at the begining */
  if( mxIsSparse(data_X)== true ) 
  {
    /* for i=1:nData, X(:,i) = X(:,i)*y(i); end*/
    for(i=0; i < nData; i++) 
    {
        mul_sparse_col(1/data_y[i], data_X, i);
    }
  }
  else
  {
    
    /* features in double precision */
    if( mxIsDouble(data_X) )
    {    
      ptr_double = mxGetPr(data_X);
      for(i=0; i < nData; i++) {
	for(j=0; j < nDim; j++ ) {
	  ptr_double[LIBOCAS_INDEX(j,i,nDim)] = ptr_double[LIBOCAS_INDEX(j,i,nDim)]/data_y[i];
	}
      }
    }

    /* features in single precision */
    if( mxIsSingle(data_X) )
    {    
      ptr_single = (float*)mxGetPr(data_X);
      for(i=0; i < nData; i++) {
	for(j=0; j < nDim; j++ ) {
	  ptr_single[LIBOCAS_INDEX(j,i,nDim)] = ptr_single[LIBOCAS_INDEX(j,i,nDim)]/data_y[i];
	}
      }
    }
    
    /* features in int8  */
    if( mxIsInt8(data_X) )
    {    
      ptr_int8 = (int8_t*)mxGetPr(data_X);
      for(i=0; i < nData; i++) {
	for(j=0; j < nDim; j++ ) {
	  ptr_int8[LIBOCAS_INDEX(j,i,nDim)] = ptr_int8[LIBOCAS_INDEX(j,i,nDim)]/data_y[i];
	}
      }
    }        
  }

  /* divide labels by Cs as it was at the begining */
  if(num_of_Cs > 1)
  {
    for(i=0; i < nData; i++) 
      data_y[i] = data_y[i]/vec_C[i];
  }

  /* create output variables */
  plhs[1] = mxCreateDoubleScalar( W0 );
  
  const char *field_names[] = {"nTrnErrors","Q_P","Q_D","nIter","nCutPlanes","exitflag",
                               "init_time","output_time","sort_time",
                               "qp_solver_time","add_time","w_time","print_time",
                               "ocas_time","total_time"}; 
  mwSize dims[2] = {1,1};  

  plhs[2] = mxCreateStructArray(2, dims, (sizeof(field_names)/sizeof(*field_names)), 
                                field_names);
  
  mxSetField(plhs[2],0,"nIter",mxCreateDoubleScalar((double)ocas.nIter));
  mxSetField(plhs[2],0,"nCutPlanes",mxCreateDoubleScalar((double)ocas.nCutPlanes));
  mxSetField(plhs[2],0,"nTrnErrors",mxCreateDoubleScalar(ocas.trn_err)); 
  mxSetField(plhs[2],0,"Q_P",mxCreateDoubleScalar(ocas.Q_P)); 
  mxSetField(plhs[2],0,"Q_D",mxCreateDoubleScalar(ocas.Q_D)); 
  mxSetField(plhs[2],0,"init_time",mxCreateDoubleScalar(init_time)); 
  mxSetField(plhs[2],0,"output_time",mxCreateDoubleScalar(ocas.output_time)); 
  mxSetField(plhs[2],0,"sort_time",mxCreateDoubleScalar(ocas.sort_time)); 
  mxSetField(plhs[2],0,"qp_solver_time",mxCreateDoubleScalar(ocas.qp_solver_time)); 
  mxSetField(plhs[2],0,"add_time",mxCreateDoubleScalar(ocas.add_time)); 
  mxSetField(plhs[2],0,"w_time",mxCreateDoubleScalar(ocas.w_time)); 
  mxSetField(plhs[2],0,"print_time",mxCreateDoubleScalar(ocas.print_time)); 
  mxSetField(plhs[2],0,"ocas_time",mxCreateDoubleScalar(ocas.ocas_time)); 
  mxSetField(plhs[2],0,"total_time",mxCreateDoubleScalar(total_time)); 
  mxSetField(plhs[2],0,"exitflag",mxCreateDoubleScalar((double)ocas.exitflag)); 

  return;
}

