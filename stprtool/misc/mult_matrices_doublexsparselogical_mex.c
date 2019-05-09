/*-------------------------------------------------------------------- 
 mult_matrices_doublexsparselogical_mex.c: computes C = A*B 
   where C [m x p double] and A [m x n double] and B [n x p sparse logical] 

 Compile:  
   mex mult_matrices_doublexsparselogical_mex.c
                                                       
 Synopsis:  
   C = mult_matrices_doublexsparselogical_mex(A,B)

 About: Statistical Pattern Recognition Toolbox, (C) 1999-2011
 (W) Vojtech Franc
-------------------------------------------------------------------- */
 
#include "mex.h"
#include "matrix.h"
#include <math.h>
#include <stdint.h>

/* ==============================================================
 Main MEX function.
============================================================== */

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[] )
{
  double *A, *C, tmp;
  mwSize n, m, p;
  mwIndex i, j, k;
  mxLogical *B_pr;
  mwIndex *B_ir, *B_jc;

  if( nrhs != 2)
    mexErrMsgTxt("Incorrect number of input arguments.\n"
                 "Synopsis: C = mult_matrices_int32xdouble_mex(A,B)\n"
                 " Input:  A [n x m (double)] \n"
                 "         B [m x p (sparse logical)]\n"
                 " Output: C [n x p (double)\n");

  A = (double*)mxGetPr( prhs[0] );
  B_pr = (mxLogical*)mxGetPr( prhs[1] );
  B_ir = mxGetIr( prhs[1] );
  B_jc = mxGetJc( prhs[1] );

  n = mxGetM( prhs[0] );
  m = mxGetN( prhs[0] );
  p = mxGetN( prhs[1] );

  if( mxGetM(prhs[1]) != m )
     mexErrMsgTxt("Matrices A and B cannot be multiplied due to their improper size.");
  
  plhs[0] = mxCreateDoubleMatrix(n,p,mxREAL);
  C = mxGetPr( plhs[0] );

  for(j=0; j < p; j++)
  {
    for(i=0; i < n; i++)
    {
      tmp = 0;
      if( B_jc[j+1] - B_jc[j] > 0)
      {
        for(k=B_jc[j]; k <= B_jc[j+1]-1; k++)
        {
          if(B_pr[k])
          {
            tmp += A[i + B_ir[k]*n];
          }
        }        
      }
      C[i+j*n] = tmp;
        
    }
  }

}

