/*-------------------------------------------------------------------- 
 mult_matrices_doublexint8_mex.c: computes C = A*B 
   where C [m x p double] and A [m x n double] and B [n x p int8] 

 Compile:  
   mex mult_matrices_doublexint8_mex.c
                                                       
 Synopsis:  
   C = mult_matrices_doublexint8_mex(A,B)

 About: Statistical Pattern Recognition Toolbox, (C) 1999-2011
 (W) Vojtech Franc
-------------------------------------------------------------------- */

#include "mex.h"
#include "matrix.h"
#include <math.h>
#include <stdlib.h>
#include <limits.h>
#include <stdint.h>

/* ==============================================================
 Main MEX function.
============================================================== */

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[] )
{

  double *A, *ptrA, *C;
  int8_t *B, *ptrB, value;
  mwSize n, m, p;
  mwIndex i, j, k, idx;

  if( nrhs != 2)
    mexErrMsgTxt("Incorrect number of input arguments.\n"
                 "Synopsis: C = mult_matrices_int32xdouble_mex(A,B)\n"
                 " Input:  A [n x m (double)] \n"
                 "         B [m x p (int8)]\n"
                 " Output: C [n x p (double)\n");

  A = mxGetPr( prhs[0] );
  B = mxGetLogicals(prhs[1]);
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
        (*C) += (*ptrA)*(double)(*ptrB);
     
        ptrA += n;
        ptrB ++;
      }
        
      C++;
    }
  }

}
