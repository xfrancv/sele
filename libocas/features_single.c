/*-----------------------------------------------------------------------
 * features_single.c: Helper functions for the OCAS solver working with 
 *                    features in single precision.
 *-------------------------------------------------------------------- */


#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "ocas_helper.h"
#include "features_single.h"

/*----------------------------------------------------------------------------------
  full_add_new_cut( new_col_H, new_cut, cut_length, nSel ) does the following:

    new_a = sum(data_X(:,find(new_cut ~=0 )),2);
    new_col_H = [full_A(:,1:nSel)'*new_a ; new_a'*new_a];
    full_A(:,nSel+1) = new_a;

  ---------------------------------------------------------------------------------*/
int full_single_add_new_cut( double *new_col_H, 
                       uint32_t *new_cut, 
                       uint32_t cut_length, 
                       uint32_t nSel,
                       void* user_data)
{
  double sq_norm_a;
  float *ptr;
  uint32_t i, j;

  ptr = (float*)mxGetPr(data_X);

  memset(new_a, 0, sizeof(double)*nDim);


  for(i=0; i < cut_length; i++) {
    for(j=0; j < nDim; j++ ) {
      new_a[j] += (double)ptr[LIBOCAS_INDEX(j,new_cut[i],nDim)];
    }

    A0[nSel] += X0*data_y[new_cut[i]];    
  }

  /* compute new_a'*new_a and insert new_a to the last column of full_A */
  sq_norm_a = A0[nSel]*A0[nSel];
  for(j=0; j < nDim; j++ ) {
    sq_norm_a += new_a[j]*new_a[j];
    full_A[LIBOCAS_INDEX(j,nSel,nDim)] = new_a[j];
  }

  new_col_H[nSel] = sq_norm_a;
  for(i=0; i < nSel; i++) {
    double tmp = A0[nSel]*A0[i];

    for(j=0; j < nDim; j++ ) {
      tmp += new_a[j]*full_A[LIBOCAS_INDEX(j,i,nDim)];
    }
    new_col_H[i] = tmp;
  }

/*  mxFree( new_a );*/

  return 0;
}


/*----------------------------------------------------------------------
  full_compute_output( output ) does the follwing:

  output = data_X'*W;
  ----------------------------------------------------------------------*/
int full_single_compute_output( double *output, void* user_data )
{
  uint32_t i, j;
  float *ptr;
  double tmp;

  ptr = (float*)mxGetPr( data_X );

  for(i=0; i < nData; i++) { 
    tmp = data_y[i]*X0*W0;

    for(j=0; j < nDim; j++ ) {
      tmp += W[j] * (double)ptr[LIBOCAS_INDEX(j,i,nDim)];
    }
    output[i] = tmp;
  }
  
  return 0;
}
