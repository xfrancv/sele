/*=================================================================
 * LBPPYR computes LBP features on scale-pyramid of input image.
 * 
 * Synopsis:
 *  F = lbppyrsparse_mat(I, P)
 * where
 *  I [H x W (uint8)] is input image.
 *  P [1 x 1 (double)] is height of the scale-pyramid.
 *  F [N x 1 (double sparse)] LBP features stucked to a column vector. 
 *
 *=================================================================*/

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <mex.h>
#include <time.h>
#include <errno.h>

#include "liblbp_v2.h"

#define MIN(A,B) ((A) > (B) ? (B) : (A))
#define MAX(A,B) ((A) < (B) ? (B) : (A))
#define ABS(A) ((A) < 0 ? -(A) : (A))
#define INDEX(ROW,COL,NUM_ROWS) ((COL)*(NUM_ROWS)+(ROW))


/*======================================================================
  Main code plus interface to Matlab.
========================================================================*/

void mexFunction( int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[] )
{

  uint8_t *p_img;
  uint16_t height_of_pyramid;
  char *p_features;
  uint32_t num_features;
  uint32_t *p_tmp_img;
  mwSize num_rows;
  mwSize num_cols;
  int i ,j, cnt;
  mxArray* sp_mat_X;
  int nnz;
  mwIndex *irs, *jcs;
  double *sr;
  uint32_t *p_index;  

  if( nrhs != 2 )
    mexErrMsgTxt("Two input arguments required.\n\n"
                 "LBPPYR computes LBP features on scale-pyramid of input image.\n"
                 "Synopsis:\n"
                 "  features = lbppyrsparse(img, heightOfpyramid)\n"
                 "where\n"
                 "  img [numRows x numCols (uint8)] is input image.\n"
                 "  heightOfPyramid [1 x 1 (double)] height (num of levels) of the scale pyramid.\n"
                 "  features [numFeatures x 1 (sparse double)] LBP features stucked to column vector. \n");

  p_img = (uint8_t*)mxGetPr(prhs[0]);
  height_of_pyramid = (uint16_t)mxGetScalar(prhs[1]);

  num_rows = mxGetM(prhs[0]);
  num_cols = mxGetN(prhs[0]);

/*  mexPrintf("num_cols: %d\nnum_rows=%d\nheight_of_pyramid=%d\n", num_cols,num_rows, height_of_pyramid);*/

  p_tmp_img = (uint32_t*)mxCalloc(num_rows*num_cols,sizeof(uint32_t));
  if(p_tmp_img == NULL) {
    mexErrMsgTxt("Not enough memory p_tmp_img.");
  }

  cnt=0;
  for(i=0; i < num_cols; i++) {
    for(j=0; j < num_rows; j++ ) {
      p_tmp_img[cnt++] = p_img[INDEX(j,i,num_rows)];
    }
  }
    
  num_features = liblbp_pyr_get_dim(num_rows, num_cols, height_of_pyramid);
  //  mexPrintf("num_features: %d\n", num_features);
  nnz = num_features / 256 ;

  p_index = (uint32_t*)mxCalloc(nnz,sizeof(uint32_t));
  if(p_index == NULL) {
    mexErrMsgTxt("Not enough memory p_index.");
  }

  liblbp_pyr_features_sparse(p_index, nnz, p_tmp_img, num_rows, num_cols);

  sp_mat_X = mxCreateSparse(num_features, 1, nnz, mxREAL);

  if( sp_mat_X == NULL)
    mexErrMsgTxt("Not enough memory to allocate sp_mat_X");
  plhs[0] = sp_mat_X;

  sr  = mxGetPr(sp_mat_X);
  irs = mxGetIr(sp_mat_X);
  jcs = mxGetJc(sp_mat_X);

  jcs[1] = nnz;

  for(i=0; i < nnz; i++)
  {
    irs[i] = p_index[i];
    sr[i] = 1.0;
  }

  mxFree(p_index);
  mxFree(p_tmp_img);

  return;
}

