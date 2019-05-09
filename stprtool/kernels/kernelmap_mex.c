/* --------------------------------------------------------------------
 kernelproj_mex.c: MEX-file code implementing kernel projection.

 Compile:  mex kernelproj_mex.c kernel_fun.c


 -------------------------------------------------------------------- */

#include "mex.h"
#include "matrix.h"
#include <math.h>
#include <stdlib.h>
#include "kernel_functions.h"

/*#define __DEBUG_KERNELMAP*/

/* ==============================================================
 Main MEX function - interface to Matlab.
============================================================== */
void mexFunction( int nlhs, mxArray *plhs[],
		  int nrhs, const mxArray *prhs[] )
{
   long i, j, k;
   long nSV, nOutDim, nObjects;
   double *Alpha;
   double *bias;
   double *proj;
   double val;

  
   /* ------------------------------------------------------------------------- */
   /*  proj = kernelmap_mex( data, Alpha, bias, SV, kernel_name, kernel_args )  */
   /* ------------------------------------------------------------------------- */
   if( nrhs != 6) 
   {
      mexErrMsgTxt("Wrong number of input arguments.\n"
                   "Synopsis:\n"
                   "  proj = kernelmap_mex(data,Alpha,bias,SV,kernelName,kernelArgs)\n"
                   "Input:\n"
                   "  data [array of N objects] Data to be projected.\n"
                   "  Alpha [nOutDims x nSV] Multipliers.\n"
                   "  bias [nOutDims x 1] Bias.\n"
                   "  SV [array of nSV objects] Support vectors.\n"
                   "  kernelName [string] Kernel identifier.\n"
                   "  kernelArgs [...] Kernel arguments.\n"
                   "Output:\n"
                   "  proj [nOutDims x N] Projected data.\n");
   }
   else
   {
      /* multipliers Alpha [nsv  x new_dim] */
      if( !mxIsNumeric(prhs[1]) || !mxIsDouble(prhs[1]) ||
        mxIsEmpty(prhs[1])    || mxIsComplex(prhs[1]) )
        mexErrMsgTxt("Input argument Alpha must be 2D matrix of doubles.");

      /* vector b [nsv  x 1] */
      if( !mxIsNumeric(prhs[2]) || !mxIsDouble(prhs[2]) ||
        mxIsEmpty(prhs[2])    || mxIsComplex(prhs[2]) )
        mexErrMsgTxt("Input b must be a real vector.");

      if( !kernel_init( prhs[0], prhs[3], prhs[4], prhs[5] ) )
      {
        mexErrMsgTxt("Kernel initialization failed.");
        return;
      }
      
     Alpha = mxGetPr(prhs[1]);
     bias = mxGetPr(prhs[2]);

     /* get data dimensions */ 
     nObjects = kernel_getM();
     nSV = kernel_getN();
     nOutDim = mxGetM(prhs[1]);

#ifdef __DEBUG_KERNELMAP
     mexPrintf("nObjects=%d, nSV=%d, nOutDim=%d\n", nObjects, nSV, nOutDim );
#endif     

     /* creates output kernel matrix. */
     plhs[0] = mxCreateDoubleMatrix(nOutDim,nObjects,mxREAL);
     proj = mxGetPr(plhs[0]);

     /* computes kernel projection */
     for( i = 0; i < nObjects; i++ ) 
     {

       for( k = 0; k < nOutDim; k++) 
         proj[ k + i*nOutDim ] = bias[k]; 

       for( j = 0; j < nSV; j++ ) 
       {
         val = kernel_eval(i,j);

         for( k = 0; k < nOutDim; k++) 
           proj[ k + i*nOutDim ] += val*Alpha[ k + j*nOutDim ]; 

       }
     }
   } 

   return;
}
