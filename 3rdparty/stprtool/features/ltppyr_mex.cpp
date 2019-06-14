/*=================================================================
 * LTPPYR computes uniform LBP features on scale-pyramid of input image.
 * 
 * Synopsis:
 *  F = ltppyr(I, P, th)
 * where
 *  I [H x W (uint8)] is input image.
 *  P [1 x 1 (double)] is height of the scale-pyramid.
 *  th [1 x 1 (uint8_t) threshold
 *  F [N x 1 (uint8)] LBP features stucked to a column vector. 
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
  uint8_t th;
  uint16_t height_of_pyramid;
  char *p_features;
  uint32_t num_features;
  uint32_t *p_tmp_img;
  mwSize num_rows;
  mwSize num_cols;
  int i ,j, cnt;

  

  if( nrhs != 3 )
    mexErrMsgTxt("Two input arguments required.\n\n"
                 "LTPPYR computes LTP features on scale-pyramid of the input image.\n"
                 "Synopsis:\n"
                 "  features = ltppyr(img, heightOfpyramid, th)\n"
                 "where\n"
                 "  img [numRows x numCols (uint8)] is input image.\n"
                 "  heightOfPyramid [1 x 1 (double)] height (num of levels) of the scale pyramid.\n"
                 "  th [1 x 1 (uint8)] threshold.\n"
                 "  features [numFeatures x 1 (uint8)] LBP features stucked to column vector. \n");

  p_img = (uint8_t*)mxGetPr(prhs[0]);
  height_of_pyramid = (uint16_t)mxGetScalar(prhs[1]);
  th = (uint8_t)mxGetScalar(prhs[2]);

  num_rows = mxGetM(prhs[0]);
  num_cols = mxGetN(prhs[0]);

/*  mexPrintf("num_cols: %d\nnum_rows=%d\nheight_of_pyramid=%d\nth=%d\n", num_cols,num_rows, height_of_pyramid, th);*/

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
    
  num_features = liblbp_pyr_get_dim(num_rows, num_cols, height_of_pyramid) * 2;

  plhs[0] = mxCreateNumericMatrix(num_features, 1, mxUINT8_CLASS, mxREAL);
  p_features = (char*)mxGetPr(plhs[0]);
  liblbp_ltp_pyr_features(p_features, num_features, p_tmp_img, num_rows, num_cols, th );
  
  mxFree(p_tmp_img);

  return;
}

