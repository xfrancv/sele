/*-------------------------------------------------------------------- 
 knnest_mex.c: finds K-NN estimate of a posterior probability.

 Compile:  knnest_mex.c
                                                       
 Synopsis:  
   out = knnest_mex(tst_X,trn_X,trn_y, K )

 Input:
  tst_X [nDim x nTst] test feature vectors.
  trn_X [nDim x nTrn] training feature vectors.
  trn_y [ nTrn x 1] training labels; must be from 1 to nY
  K [1 x 1] number of the nearest neighbours
 Output:
  out [nY x n_tst] out(y,i) is the number of training vectors with 
    label y found among K nearest neighbors of the vector tst_X(:,i).

 About: (c) Statistical Pattern Recognition Toolbox, (C) 1999-2010
 Written by Vojtech Franc
 <a href="http://www.cvut.cz">Czech Technical University Prague</a>,
 <a href="http://www.feld.cvut.cz">Faculty of Electrical engineering</a>,
 <a href="http://cmp.felk.cvut.cz">Center for Machine Perception</a>

-------------------------------------------------------------------- */
 
#include "mex.h"
#include "matrix.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdint.h>

#define MAX(A,B)   (((A) > (B)) ? (A) : (B) )
#define MIN(A,B)   (((A) < (B)) ? (A) : (B) ) 
 

/* ==============================================================
 Main MEX function.
============================================================== */
static void compute_norm2(double * norm2_vec, const mxArray * matrix);

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[] )
{
  double *tst_feat, *trn_feat, *trn_labels, *dist, *norm2_trn, *norm2_tst, *out;
  int K, nDim, i, j, l, nY, nTrn, nTst;


  /* get input arguments */

  if( nrhs < 3 || nrhs > 4)
    mexErrMsgTxt("Incorrect number of input arguments.\n"
                 "Synopsis: out = knnest( tst_X, trn_X, trn_y, K )\n"
                 "          out = knnest( tst_X, trn_X, trn_y)\n"
                 " Input: tst_X [nDim x nTst] testing examples.\n"
                 "        trn_X [nDim x nTrn] training examples vectors.\n"
                 "        trn_y [ nTrn x 1] training labels; must be from 1 to nY .\n"
                 "        K [1 x 1] number of the nearest neighbours (default 1).\n"
                 " Output: out [nY x n_tst] out(y,i) is the number of training examples \n"
                 "        with label y found among K nearest neighbors of tst_X(:,i).\n");

  tst_feat = mxGetPr( prhs[0] );
  trn_feat = mxGetPr( prhs[1] );
  nDim = mxGetM(prhs[0]);
  nTst = mxGetN(prhs[0]);
  nTrn = mxGetN(prhs[1]);
  trn_labels = mxGetPr(prhs[2]);  
  
  if( mxIsSparse(prhs[2]))
     mexErrMsgTxt("y can't be sparse.");

  if( mxIsSparse(prhs[0]) ^ mxIsSparse(prhs[1]))
     mexErrMsgTxt("trn_X and tst_X must be same type (sparse / dense).");

  if( mxIsLogical(prhs[0]) || mxIsLogical(prhs[1]) || mxIsLogical(prhs[2]))
     mexErrMsgTxt("The input arguments must be double.");
    
  if( nDim != mxGetM( prhs[1] )) 
     mexErrMsgTxt("Dimension of training and testing must be the same.");
  if( nrhs >= 4)
    K = (int)mxGetScalar(prhs[3]);    
  else
    K = 1;

  for(nY = 0, i=0; i < nTrn; i++)
    nY = MAX(nY, trn_labels[i]);

  /*  mexPrintf("K=%d\n"
            "nY=%d\n"
            "nTrn=%d\n"
            "nTst=%d\n"
            "nDims=%d\n", K, nY, nTrn, nTst, nDim);
  */
   
  /*  allocate output */
  plhs[0] = mxCreateDoubleMatrix(nY,nTst,mxREAL);
  out = mxGetPr( plhs[0] );

  /*--------------------------*/

  if( (dist = mxCalloc(nTrn, sizeof(double))) == NULL)
      mexErrMsgTxt("Not enough memory for vector dist.");

  if( (norm2_trn = mxCalloc(nTrn, sizeof(double))) == NULL)
      mexErrMsgTxt("Not enough memory for vector dist.");

  if( (norm2_tst = mxCalloc(nTst, sizeof(double))) == NULL)
      mexErrMsgTxt("Not enough memory for vector dist.");

  double *ptr1, *ptr2;
  double tmp = 0;

  compute_norm2(norm2_tst, prhs[0]);
  compute_norm2(norm2_trn, prhs[1]);
  bool sparse = mxIsSparse(prhs[0]);
  
  for( i=0; i < nTst; i++ ) 
  {
    if(sparse) {
      mwIndex *irs1 = mxGetIr(prhs[0]), 
	      *jcs1 = mxGetJc(prhs[0]),
	      *irs2 = mxGetIr(prhs[1]), 
	      *jcs2 = mxGetJc(prhs[1]);   
      int l2;
      for( j=0; j < nTrn; j++ ) 
      {
	tmp = 0;
	l = jcs1[i];
	l2 = jcs2[j];
	while(l<jcs1[i+1] && l2<jcs2[j+1]) {
	  if(irs1[l]==irs2[l2]) {
	    tmp += tst_feat[l]*trn_feat[l2];
	    l++; l2++;
	  } else {
	    if(irs1[l]>irs2[l2])
	      l2++;
	    else
	      l++;
	  }
	}
	dist[j] = norm2_trn[j] - 2*tmp + norm2_tst[i];
      }
    } else {
      for( j=0; j < nTrn; j++ ) 
      {
	tmp = 0;
	ptr1 = tst_feat + i*nDim;
	ptr2 = trn_feat + j*nDim;

	for( l=0; l < nDim; l++ ) 
	{
	  tmp += (*ptr1)*(*ptr2);
	  ptr1++;
	  ptr2++;
	}
	    
	dist[j] = norm2_trn[j] - 2*tmp + norm2_tst[i];
      }
    }
    for( l=0; l < K; l++) 
    {
      double min_dist = mxGetInf();
      uint32_t min_idx;
      for( j=0; j < nTrn; j++ ) 
      {
        if( min_dist > dist[j] ) 
        {
          min_idx = j;
          min_dist = dist[j];
        }
      }
      dist[ min_idx ] = mxGetInf();
      out[ (uint32_t)trn_labels[min_idx] + nY*i - 1] ++;
    }    
  }
  
  mxFree( dist );  

  mxFree( norm2_tst );  
  mxFree( norm2_trn ); 
}

static void compute_norm2(double * norm2_vec, const mxArray * matrix) {
  int nDim = mxGetM(matrix), 
      nData = mxGetN(matrix);
  double *feat = mxGetPr(matrix);
  double tmp;
  int i, l;

  if(mxIsSparse(matrix)) {
    mwIndex *jcs = mxGetJc(matrix);
    for(i = 0; i < nData; i++)
    {
	tmp = 0;
	for(l = jcs[i]; l<jcs[i+1]; l++)
	  tmp += feat[l]*feat[l];
	norm2_vec[i] = tmp;
    }
  } else {
    double *ptr;
    for( i=0; i < nData; i++ )
    {
	tmp = 0;
	ptr = feat + i*nDim;
	for( l=0; l < nDim; l++ )
	{
	  tmp += (*ptr) * (*ptr);
	  ptr++;
	}
	norm2_vec[i]  = tmp;
    }
  }
}
