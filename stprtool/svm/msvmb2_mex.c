/*---------------------------------------------------------------------------
 msvmb2.c: MEX training multi-class SVM classifier with L2-soft margin and 
           L2-regularized bias terms.

 Compile: 
  mex msvmb2_mex.c libwp_splx.c kernel_functions.c

 Synopsis:
  [Alpha,bias,exitflag,nKerEvals,nKerCols,nIter,QP,QD] = 
     msvmb2(X,y,kernel_name,kernel_args,C,MaxIter,TolAbs,TolRel,QP_TH,CacheSize,verb)

 Input:
  X [...] Training inputs.
  y [nExamples x 1] Labels (1,2,...,nY).
  kernel_name [string] Kernel identifier.
  kernel_args [...] Kernel argument.
  C [1x1] Regularization constant.
  MaxIter [1x1] Maximal number of iterations.
  TolAbs [1x1] Absolute tolerance stopping condition.
  TolRel [1x1] Relative tolerance stopping condition.
  QP_TH [1x1] Threshold on the lower bound.
  CacheSize [1x1] Number of columns of kernel matrix to be cached.
    It takes CacheSize*nExamples*size(double) bytes of memory.
  verb [1x1] If 1 print some info.

 Output:
  Alpha [nY x nExamples] Dual weights.
  bias [nYx1] Bias.
  exitflag [1x1] Indicates which stopping condition was used:
       0  ... Maximal number of iterations reached: nIter >= MaxIter.
       1  ... Relative tolerance reached: QP-QD <= abs(QP)*TolRel
       2  ... Absolute tolerance reached: QP-QD <= TolAbs
       3  ... Objective value reached threshold: QP <= QP_TH.
  nKerEvals [1x1] Number of kernel evaluations.
  nKerCols [1x1] Number of requested columns of virtual kernel matrix.
  trn_err [1x1] Training error.
  nIter [1x1] Number of iterations of the QP solver.
  QP [1x1] Primal objective.
  QD [1x1] Dual objective.

-------------------------------------------------------------------- */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "mex.h"
#include "matrix.h"
#include "kernel_functions.h"
#include "libqp.h"

/*#define __DEBUG_MSVMB2*/

#define KDELTA(A,B) (A==B)
#define KDELTA4(A1,A2,A3,A4) ((A1==A2)||(A1==A3)||(A1==A4)||(A2==A3)||(A2==A4)||(A3==A4))

unsigned long nKerCols;
long nClasses;      
unsigned long nVirtExamples;        /* number of virtual examples (of single-class problem) */
long nExamples;                     /* number of input training examples */
double diag_element;                /* regularization constant */
double *labels;                     /* Pointer to labels */

long CacheSize;                     /* number of cached columns (min 1) */

/* FIFO cache for columns of the kernel matrix */
long *cache_index;                  /* indices cached of kernel columns */
long first_kernel_inx;              /* index of first inserted column */
double **kernel_columns;            /* pointers at cached columns */

/* cache for three columns of the virtual kernel matrix */
int first_virt_inx;                 /* index of first used column */
double *virt_columns[3];            /* cache for three columns*/

long verb;

void print_state(libqp_state_T state)
{
  double tolrel = 0;

  if( (state.nIter % verb == 0) || (state.exitflag != 100))
  {
    if( state.QP != 0) tolrel = (state.QP-state.QD)/LIBQP_ABS(state.QP); 

    mexPrintf("%4d: P=%.9f D=%.9f P-D=%.9f (P-D)/|P|=%.9f\n",
              state.nIter, state.QP, state.QD, state.QP-state.QD, tolrel );
  }
}


/* ------------------------------------------------------------
  Returns pointer at a-th column of the kernel matrix.
  This function maintains FIFO cache of kernel columns.
------------------------------------------------------------ */
void *get_kernel_col( long a ) 
{
  double *col_ptr;
  long i;
  long inx;

  inx = -1;
  for( i=0; i < CacheSize; i++ ) 
  {
    if( cache_index[i] == a ) 
    { 
      inx = i; 
      break; 
    }
  }
    
  if( inx != -1 ) 
  {
    col_ptr = kernel_columns[inx];
    return( col_ptr );
  }
   
  col_ptr = kernel_columns[first_kernel_inx];
  cache_index[first_kernel_inx] = a;

  first_kernel_inx++;
  if( first_kernel_inx >= CacheSize ) 
    first_kernel_inx = 0;

  for( i=0; i < nExamples; i++ ) 
    col_ptr[i] = kernel_eval(i,a);

  return( col_ptr );
}

/* ------------------------------------------------------------
  Computes index of input example and its class label from 
  index of virtual "single-class" example.
------------------------------------------------------------ */
void get_indices2( long *index, long *class, long i )
{
   *index = i / (nClasses-1); 
   *class = (i % (nClasses-1))+1;

   if( *class >= labels[ *index ]) 
     (*class)++;

   return;
}

/* ------------------------------------------------------------
  Retures (a,b)-th element of the virtual kernel matrix 
  of size [num_virt_data x num_virt_data]. 
------------------------------------------------------------ */
double kernel_fce( long a, long b )
{
  double value;
  long i1,c1,i2,c2;

  get_indices2( &i1, &c1, a );
  get_indices2( &i2, &c2, b );

  if( KDELTA4(labels[i1],labels[i2],c1,c2) ) 
  {
    value = (+KDELTA(labels[i1],labels[i2]) 
             -KDELTA(labels[i1],c2)
             -KDELTA(labels[i2],c1)
             +KDELTA(c1,c2)
            )*(kernel_eval( i1, i2 )+1);
  }
  else
  {
    value = 0;
  }

  if(a==b) 
    value += diag_element; 

  return( value );
}

/* ------------------------------------------------------------
  Returns pointer at the a-th column of the virtual K matrix.

  (note: the b-th column must be preserved in the cache during 
   updating but b is from (a(t-2), a(t-1)) where a=a(t) and
   thus FIFO with three columns does not have to take care od b.)
------------------------------------------------------------ */
const double *get_col( uint32_t a)
{
  long i;
  long inx;
  long min_usage; 
  double *col_ptr;
  double *ker_ptr;
  double value;
  long i1,c1,i2,c2;

  nKerCols++;

  col_ptr = virt_columns[first_virt_inx++];
  if( first_virt_inx >= 3 ) 
    first_virt_inx = 0;

  get_indices2( &i1, &c1, a );
  ker_ptr = (double*) get_kernel_col( i1 );

  for( i=0; i < nVirtExamples; i++ ) 
  {
    get_indices2( &i2, &c2, i );

    if( KDELTA4(labels[i1],labels[i2],c1,c2) ) 
    {
      value = (+KDELTA(labels[i1],labels[i2]) 
               -KDELTA(labels[i1],c2)
               -KDELTA(labels[i2],c1)
               +KDELTA(c1,c2)
              )*(ker_ptr[i2]+1);
    }
    else
    {
      value = 0;
    }

    if(a==i) 
      value += diag_element; 

    col_ptr[i] = value;
  }
  
  return( col_ptr );
}


/* -------------------------------------------------------------------
 Main MEX function - interface to Matlab.
-------------------------------------------------------------------- */
void mexFunction( int nlhs, mxArray *plhs[],int nrhs, const mxArray*prhs[] )
{
  int exitflag;          /* output arg */
  long i,j ;             /* common use loop variables */
  long inx1, inx2;   
  unsigned long MaxIter; /* input arg - max number of iterations */ 
  double QP_TH;          /* input arg - threshold on lower bound */
  double C;              /* input arg - regularization const */
  double TolRel;         /* input arg */
  double TolAbs;         /* input arg */
  double trnerr;         /* output arg */
  double *tmp_ptr1;
  double *tmp_ptr2; 
  double *vector_c;      /* auxiliary */ 
  double *Alpha;         /* solution vector */ 
  double *diagK;         /* cache for diagonal of virtual K matrix */

  libqp_state_T state;
  double b;
  uint32_t *I;
  uint8_t S;

  /*------------------------------------------------------------------- */
  /* Get input arguments                                                */
  /*------------------------------------------------------------------- */
  if( nrhs != 11) 
  {
    mexErrMsgTxt("Incorrect number of input arguments.\n"
                 "Synopsis:\n"
                 "[Alpha,bias,exitflag,nKerEvals,nKerCols,nIter,QP,QD] = ... \n"
                 "   msvmb2(X,y,C,kernel_name,kernel_args,...\n"
                 "          MaxIter,TolAbs,TolRel,QP_TH,CacheSize,verb)\n"
                 "Input:\n"
                 " X [...] Training inputs.\n"
                 " y [nExamples x 1] Labels (1,2,...,nY).\n"
                 " C [1x1] Regularization constant.\n"
                 " kernel_name [string] Kernel identifier.\n"
                 " kernel_args [...] Kernel argument.\n"
                 " MaxIter [1x1] Maximal number of iterations.\n"
                 " TolAbs [1x1] Absolute tolerance stopping condition.\n"
                 " TolRel [1x1] Relative tolerance stopping condition.\n"
                 " QP_TH [1x1] Threshold on primal objective.\n"
                 " CacheSize [1x1] Number of columns of kernel matrix to be cached.\n"
                 "   It takes cache*nExamples*size(double) bytes of memory.\n"
                 " verb [1x1] If 1 print some info.\n"
                 "Output:\n"
                 " Alpha [nY x nExamples] Dual weights.\n"
                 " bias [nYx1] Bias.\n"
                 " exitflag [1x1] Indicates which stopping condition was used:\n"
                 "   0  ... Maximal number of iterations reached: nIter >= MaxIter.\n"
                 "   1  ... Relative tolerance reached: QP-QD <= abs(QP)*TolRel\n"
                 "   2  ... Absolute tolerance reached: QP-QD <= TolAbs\n"
                 "   3  ... Objective value reached threshold: QP <= QP_TH.\n"
                 " nKerEvals [1x1] Number of kernel evaluations.\n"
                 " nKerCols [1x1] Number of requested columns of virtual kernel matrix.\n"
                 " trn_err [1x1] Training error.\n"
                 " nIter [1x1] Number of iterations of the QP solver.\n"
                 " QP [1x1] Primal objective.\n"
                 " QD [1x1] Dual objective.\n");
  }

#ifdef __DEBUG_MSVMB2
  mexPrintf("Kernel initialization.\n");
#endif

  if( !kernel_init( prhs[0], prhs[0], prhs[3], prhs[4] ) )
  {
    mexErrMsgTxt("Kernel initialization failed.");
    return;
  }
       
#ifdef __DEBUG_MSVMB2
  mexPrintf("Setting up the QP task.\n");
#endif

  nExamples = kernel_getN();  /* get number of objects */
  labels = mxGetPr(prhs[1]);  /* pointer at data labels */
  C = mxGetScalar(prhs[2]);   /* regularization constant */
  MaxIter = mxIsInf( mxGetScalar(prhs[5])) ? LONG_MAX : (long)mxGetScalar(prhs[5]); 
  TolAbs = mxGetScalar(prhs[6]);   
  TolRel = mxGetScalar(prhs[7]);   
  QP_TH = mxIsInf( mxGetScalar(prhs[8])) ? mxGetInf() : (double)mxGetScalar(prhs[8]); 
  CacheSize = (long)mxGetScalar(prhs[9]);  
  if( CacheSize < 1 ) mexErrMsgTxt("Cache must be greater than 1."); 
  if( CacheSize > nExamples ) CacheSize = nExamples; 
  verb = (long)mxGetScalar(prhs[10]);  /* verbosity on/off */

  if(verb > 0)
    mexPrintf("Settings:\n"
            "nExamples: %d\n"
            "C: %f\n"
            "MaxIter: %ld\n"
            "TolAbs: %f\n"
            "TolRel: %f\n"
            "QP_TH: %f\n"
            "CacheSize: %d\n"
            "verb: %d\n",
            nExamples,C, MaxIter, TolAbs,TolRel, QP_TH, CacheSize, verb);


  /*------------------------------------------------------------------- */
  /* Setup QP problem, allocate kernel cache, etc.                      */
  /*------------------------------------------------------------------- */

  /* constant added to diagonal of separable problem */
  if( C!=0 ) diag_element = 1/(2*C); else diag_element = 0;

  /* get number of labels */
  nClasses = -LONG_MAX; 
  for( i = 0; i < nExamples; i++ ) 
  { 
     if( labels[i] > nClasses ) nClasses = (long)labels[i]; 
  }

  /* computes number of virtual "single-class" examples */
  nVirtExamples = (nClasses-1)*nExamples;

  nKerCols = 0;  /* counter for access to the kernel matrix */

  /* allocattes and precomputes diagonal of virtual K matrix */
  diagK = mxCalloc(nVirtExamples, sizeof(double));
  if( diagK == NULL ) mexErrMsgTxt("Not enough memory.");
  for(i = 0; i < nVirtExamples; i++ ) 
    diagK[i] = kernel_fce(i,i);

  /* allocates memory for kernel cache */
  kernel_columns = mxCalloc(CacheSize, sizeof(double*));
  if( kernel_columns == NULL ) mexErrMsgTxt("Not enough memory.");
  cache_index = mxCalloc(CacheSize, sizeof(double));
  if( cache_index == NULL ) mexErrMsgTxt("Not enough memory.");

  for(i = 0; i < CacheSize; i++ ) 
  {
    kernel_columns[i] = mxCalloc(nExamples, sizeof(double));
    if(kernel_columns[i] == NULL) mexErrMsgTxt("Not enough memory.");

    cache_index[i] = -2;
  }
  first_kernel_inx = 0;

  /* allocates memory for three virtual kernel matrix columns */
  for(i = 0; i < 3; i++ ) 
  {
    virt_columns[i] = mxCalloc(nVirtExamples, sizeof(double));
    if(virt_columns[i] == NULL) 
      mexErrMsgTxt("Not enough memory for cache of virtual examples.");
  }
  first_virt_inx = 0; 

  /* Solution vector */
  Alpha = mxCalloc(nVirtExamples, sizeof(double));
  if( Alpha == NULL ) 
    mexErrMsgTxt("Not enough memory for dual variables.");
  Alpha[0] = 1;  

  /* Vector c; for this problem set to zero */
  vector_c = mxCalloc(nVirtExamples, sizeof(double));
  if( vector_c == NULL ) 
    mexErrMsgTxt("Not enough memory for vector_c.");

  I = (uint32_t*)mxCalloc(nVirtExamples, sizeof(uint32_t));
  if(I == NULL) mexErrMsgTxt("Not enough memory for I.");
  for(i=0; i < nVirtExamples; i++)
    I[i] = 1;

  b = 1;
  S = 0;

  /*------------------------------------------------------------------- */
  /* Call QP solver                                                     */
  /*------------------------------------------------------------------- */
#ifdef __DEBUG_MSVMB2
  mexPrintf("Running QP solver.\n");
#endif


  if( verb == 0)
  {
    state = libqp_splx_solver( &get_col, diagK, vector_c, &b, I,
                               &S, Alpha, nVirtExamples, MaxIter, 
                               TolAbs, TolRel, QP_TH, NULL);
  }
  else
  {
    state = libqp_splx_solver( &get_col, diagK, vector_c, &b, I,
                               &S, Alpha, nVirtExamples, MaxIter, 
                               TolAbs, TolRel, QP_TH, &print_state);
  }
    


  /*------------------------------------------------------------------- */
  /* Set outputs                                                        */
  /*------------------------------------------------------------------- */

#ifdef __DEBUG_MSVMB2
  mexPrintf("Processing outputs.\n");
#endif

  /* matrix Alpha [nClasses x nExamples] */
  plhs[0] = mxCreateDoubleMatrix(nClasses,nExamples,mxREAL);
  tmp_ptr1 = mxGetPr(plhs[0]);

  /* bias [nClasses x 1] */
  plhs[1] = mxCreateDoubleMatrix(nClasses,1,mxREAL);
  tmp_ptr2 = mxGetPr(plhs[1]);

  for( i=0; i < nClasses; i++ ) 
  {
    for( j=0; j < nVirtExamples; j++ ) 
    {
       get_indices2( &inx1, &inx2, j );

       tmp_ptr1[(inx1*nClasses)+i] += Alpha[j]*(KDELTA(labels[inx1],i+1)-KDELTA(i+1,inx2));
       tmp_ptr2[i] += Alpha[j]*(KDELTA(labels[inx1],i+1)-KDELTA(i+1,inx2));
    }
  }

  /* exitflag [1x1] */
  plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL);
  *(mxGetPr(plhs[2])) = (double)state.exitflag;

  /* nKerEvals [1x1] */
  plhs[3] = mxCreateDoubleMatrix(1,1,mxREAL);
  *(mxGetPr(plhs[3])) = (double)kernel_getCounter();

  /* nKerCols [1x1] */
  plhs[4] = mxCreateDoubleMatrix(1,1,mxREAL);
  *(mxGetPr(plhs[4])) = (double)nKerCols;

  /* nIter [1x1] */
  plhs[5] = mxCreateDoubleMatrix(1,1,mxREAL);
  *(mxGetPr(plhs[5])) = (double)state.nIter;

  /* UB [1x1] */
  plhs[6] = mxCreateDoubleMatrix(1,1,mxREAL);
  *(mxGetPr(plhs[6])) = state.QP;

  /* LB [1x1] */
  plhs[7] = mxCreateDoubleMatrix(1,1,mxREAL);
  *(mxGetPr(plhs[7])) = state.QD;

  /*------------------------------------------------------------------- */
  /* Free used memory                                                   */
  /*------------------------------------------------------------------- */

  kernel_destroy();

  mxFree( I );
  mxFree( vector_c );
  mxFree( Alpha );
  mxFree( diagK );
  for(i = 0; i < CacheSize; i++ ) 
    mxFree(kernel_columns[i]);
  for(i = 0; i < 3; i++ ) 
    mxFree(virt_columns[i]);
  mxFree( kernel_columns );
  mxFree( cache_index );

}

