/*-----------------------------------------------------------------------
 * ocas_helper.c: Implementation of helper functions for the OCAS solver.
 *
 * It supports both sparse and dense matrices and loading data from
 * the SVM^light format.
 *-------------------------------------------------------------------- */

#define _FILE_OFFSET_BITS  64

#include <pthread.h>

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>
#include <stdlib.h>
#include <time.h>

#include "libocas.h"
#include "ocas_helper.h"


/*----------------------------------------------------------------------
  full_bool_compute_output( output ) does the follwing:

  output = data_X'*W;
  ----------------------------------------------------------------------*/
int full_bool_compute_output( double *output, void* user_data )
{
  uint32_t i, j, k, dataDim; /* k is new index for the vector W */
  double tmp;
  uint8_t* ptr;

  ptr = (uint8_t*)mxGetPr( data_X );
  
  dataDim = nDim/8;

  for(i=0; i < nData; i++) { 
    tmp = X0*W0;
    
    k = 0;

    for(j=0; j < dataDim; j++ ) {
      if (ptr[LIBOCAS_INDEX(j,i,dataDim)] & 0x01) {
	tmp += W[k];
      }
      k++;
      if (ptr[LIBOCAS_INDEX(j,i,dataDim)] & 0x02) {
	tmp += W[k];
      }
      k++;
      if (ptr[LIBOCAS_INDEX(j,i,dataDim)] & 0x04) {
	tmp += W[k];
      }
      k++;
      if (ptr[LIBOCAS_INDEX(j,i,dataDim)] & 0x08) {
	tmp += W[k];
      }
      k++;
      if (ptr[LIBOCAS_INDEX(j,i,dataDim)] & 0x10) {
	tmp += W[k];
      }
      k++;
      if (ptr[LIBOCAS_INDEX(j,i,dataDim)] & 0x20) {
	tmp += W[k];
      }
      k++;
      if (ptr[LIBOCAS_INDEX(j,i,dataDim)] & 0x40) {
	tmp += W[k];
      }
      k++;
      if (ptr[LIBOCAS_INDEX(j,i,dataDim)] & 0x80) {
	tmp += W[k];
      }
      k++;
    }
    output[i] = tmp*data_y[i];
  }
  
  return 0;
}



/*----------------------------------------------------------------------------------
  full_bool_add_new_cut( new_col_H, new_cut, cut_length, nSel ) does the following:

    new_a = sum(data_X(:,find(new_cut ~=0 )),2);
    new_col_H = [full_A(:,1:nSel)'*new_a ; new_a'*new_a];
    full_A(:,nSel+1) = new_a;

  ---------------------------------------------------------------------------------*/
int full_bool_add_new_cut( double *new_col_H, 
                           uint32_t *new_cut, 
                           uint32_t cut_length, 
                           uint32_t nSel,
                           void* user_data)
{
  /*****************************/
  double sq_norm_a;
  uint32_t i, j, k, dataDim; /* k is new index for the vector a*/
  uint8_t *ptr;

  ptr = (uint8_t*)mxGetPr(data_X);
  
  dataDim = nDim/8;
  
  memset(new_a, 0, sizeof(double)*nDim);


  for(i=0; i < cut_length; i++) {
    
    k = 0;
    
    for(j=0; j < dataDim; j++ ) {
      if (ptr[LIBOCAS_INDEX(j,new_cut[i],dataDim)] & 0x01) {
	new_a[k] += data_y[new_cut[i]];
      }
      k++;
      if (ptr[LIBOCAS_INDEX(j,new_cut[i],dataDim)] & 0x02) {
	new_a[k] += data_y[new_cut[i]];
      }
      k++;
      if (ptr[LIBOCAS_INDEX(j,new_cut[i],dataDim)] & 0x04) {
	new_a[k] += data_y[new_cut[i]];
      }
      k++;
      if (ptr[LIBOCAS_INDEX(j,new_cut[i],dataDim)] & 0x08) {
	new_a[k] += data_y[new_cut[i]];
      }
      k++;
      if (ptr[LIBOCAS_INDEX(j,new_cut[i],dataDim)] & 0x10) {
	new_a[k] += data_y[new_cut[i]];
      }
      k++;
      if (ptr[LIBOCAS_INDEX(j,new_cut[i],dataDim)] & 0x20) {
	new_a[k] += data_y[new_cut[i]];
      }
      k++;
      if (ptr[LIBOCAS_INDEX(j,new_cut[i],dataDim)] & 0x40) {
	new_a[k] += data_y[new_cut[i]];
      }
      k++;
      if (ptr[LIBOCAS_INDEX(j,new_cut[i],dataDim)] & 0x80) {
	new_a[k] += data_y[new_cut[i]];
      }
      k++;
    }

    A0[nSel] += X0*data_y[new_cut[i]];    
  }

  /*****************************/

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


