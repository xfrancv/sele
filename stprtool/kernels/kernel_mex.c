/* --------------------------------------------------------------------
 kernel.c: MEX-function which evaluates kernel function for given objects.

 Compile:  mex kernel_mex.c kernel_functions.c

 Written (W) 1999-2010, Written by Vojtech Franc
 Copyright (C) 1999-2010, Czech Technical University in Prague
 -------------------------------------------------------------------- */


#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "mex.h"
#include "matrix.h"
#include "kernel_functions.h"


/* ==============================================================
 Main MEX function - interface to Matlab.
============================================================== */
void mexFunction( int nlhs, mxArray *plhs[],
	              int nrhs, const mxArray *prhs[] )
{
   long i, j, N, M;
   double tmp;
   double *K;
   
   /************************************/
   /* K = kernel( data, kernel_name, kernel_args )     */
   /************************************/
   if( nrhs == 3)
   {

     if( !kernel_init( prhs[0], prhs[0], prhs[1], prhs[2] ) )
     {
       mexErrMsgTxt("Kernel initialization failed.");
       return;
     }
       
     N = kernel_getN();  /* get number of objects */
     
     /* creates output kernel matrix */
     plhs[0] = mxCreateDoubleMatrix(N,N,mxREAL);
     K = mxGetPr(plhs[0]);

     /* evaluate kernels */ 
     for( i = 0; i < N; i++ ) 
     {
        for( j = i; j < N; j++ ) 
        {
           tmp = kernel_eval( i, j );
           K[i*N+j] = tmp; 
           K[j*N+i] = tmp; /* kernel matrix is symetric */
        }
     }
   } 
   
   /*******************************************/
   /* K = kernel( dataA, dataB, kernel_name, kernel_args )    */
   /*******************************************/
   else if( nrhs == 4)
   {
     if( !kernel_init( prhs[0], prhs[1], prhs[2], prhs[3] ) )
     {
       mexErrMsgTxt("Kernel initialization failed.");
       return;
     }
       
     M = kernel_getM();  /* get number of objects in prhs[0] */
     N = kernel_getN();  /* get number of objects in prhs[1] */
     
     /* creates output kernel matrix */
     plhs[0] = mxCreateDoubleMatrix(M,N,mxREAL);
     K = mxGetPr(plhs[0]);

     /* evaluate kernels */
     for( i = 0; i < N; i++ ) 
     {
       for( j = 0; j < M; j++ ) 
       {
         K[i*M+j] = kernel_eval( j, i );
       }
     }
   }
   else
   {
      mexErrMsgTxt("Wrong number of input arguments.\n"
                   "KERNEL evaluates kernel function for given objects \n"
                   "\n"
                   "K = kernel( data, kernelName, kernelArgs )\n"
                   "Input:\n"
                   "  data [array of N matlab objects]\n"
                   "  kernelName [string] Kernel identifier.\n"
                   "  kernelArgs [matlab object] Kernel arguments.\n"
                   "Output:\n"
                   "  K [N x N] Kernel matrix; K(i,j) = kernel(data(i),data(j)) \n"
                   "\n"
                   "K = kernel( dataA, dataB, kernelName, kernelArgs )\n"
                   "Input:\n"
                   "  dataA [array of M matlab objects] \n"
                   "  dataB [array of N matlab objects]\n"
                   "  kernelName [string] Kernel identifier.\n"
                   "  kernelArgs [matlab object] Kernel iarguments.\n"
                   "Output:\n"
                   "  K [M x N] Kernel matrix; K(i,j) = kernel(dataA(i),dataB(j))\n"
                   "\n"
                   "Implemented kernels:\n"
                   "  kernelName  kernelArgs\n"
                   "  linear       []                    x'*y\n"
                   "  rbf          [gamma]               exp(-gamma*||x-y||^2)\n"
                   "  poly         [degree arg0 arg1]    (arg1 * x'*y + arg0)^degree\n"
                   "  sigmoid      [arg0 arg1]           tanh( arg1 * x'*y + arg0 )\n"
                   "  precomputed  []                    copy of input matrix\n");
      return;
   }

   kernel_destroy();

   return;
}
