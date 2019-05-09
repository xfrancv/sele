/*-------------------------------------------------------------------- 
 mult_matrices_int32xint32_mex.c: computes C = A*B 
   where C [m x p double] and A [m x n int32] and B [n x p int32] 

 Compile:  
   mex mult_matrices_int32xint32_mex.c
                                                       
 Synopsis:  
   C = mult_matrices_int32xint32_mex(A,B)

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

  int32_t *A, *B, *ptrA, *ptrB;
  double *C;
  mwSize n, m, p;
  mwIndex i, j, k;

  if( nrhs != 2)
    mexErrMsgTxt("Incorrect number of input arguments.\n"
                 "Synopsis: C = mult_matrices_int32xint32_mex(A,B)\n"
                 " Input:  A [n x m (int32)] \n"
                 "         B [m x p (int32)]\n"
                 " Output: C [n x p (double)\n");

  A = (int32_t*)mxGetPr( prhs[0] );
  B = (int32_t*)mxGetPr( prhs[1] );
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
      ptrB = B + j*m;
      ptrA = A + i;

      for(k=0; k < m; k++)
      {
        (*C) += (double)(*ptrA) * (double)(*ptrB);
     
        ptrA += n;
        ptrB ++;
      }
        
      C++;
    }
  }

}
