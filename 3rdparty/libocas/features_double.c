/*-----------------------------------------------------------------------
 * features_double.c: Helper functions for the OCAS solver working with 
 *                    features in double precision.
 *-------------------------------------------------------------------- */


#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "ocas_helper.h"
#include "features_double.h"

/*----------------------------------------------------------------------------------
  sparse_add_new_cut( new_col_H, new_cut, cut_length, nSel ) does the following:

    new_a = zeros(nDim,nY);
    for i=1:nData
       if new_cut(i) ~= data_y(i)
          new_a(:,data_y(i)) = new_a(:,data_y(i)) + X(;,i);
          new_a(:,new_cut(i)) = new_a(:,new_cut(i)) - X(;,i);
       end
    end

    new_col_H = [sparse_A(:,1:nSel)'*new_a ; new_a'*new_a];
    sparse_A(:,nSel+1) = new_a;

    Warning: data_y is 1-based while new_cut is 0-based

  ---------------------------------------------------------------------------------*/
int msvm_sparse_add_new_cut( double *new_col_H, 
                         uint32_t *new_cut, 
                         uint32_t nSel, 
                         void* user_data )
{
/*  double *new_a, */
  double sq_norm_a;
  uint32_t i, j, nz_dims, ptr, y;

  memset(new_a, 0, sizeof(double)*nY*nDim);
  
  for(i=0; i < nData; i++)
  {
    y = (uint32_t)(data_y[i]-1);
    if(new_cut[i] != y)
    {
      add_sparse_col(&new_a[nDim*y], data_X, i);
      subtract_sparse_col(&new_a[nDim*(uint32_t)new_cut[i]], data_X, i);
    }
  }
 
  /* compute new_a'*new_a and count number of non-zero dimensions */
  nz_dims = 0; 
  sq_norm_a = 0;
  for(j=0; j < nY*nDim; j++ ) {
    if(new_a[j] != 0) {
      nz_dims++;
      sq_norm_a += new_a[j]*new_a[j];
    }
  }

  /* sparsify new_a and insert it to the last column  of sparse_A */
  sparse_A.nz_dims[nSel] = nz_dims;
  if(nz_dims > 0) {
    sparse_A.index[nSel] = NULL;
    sparse_A.value[nSel] = NULL;
    sparse_A.index[nSel] = mxCalloc(nz_dims,sizeof(uint32_t));
    sparse_A.value[nSel] = mxCalloc(nz_dims,sizeof(double));
    if(sparse_A.index[nSel]==NULL || sparse_A.value[nSel]==NULL)
    {
/*      mexErrMsgTxt("Not enough memory for vector sparse_A.index[nSel], sparse_A.value[nSel].");*/
      mxFree(sparse_A.index[nSel]);
      mxFree(sparse_A.value[nSel]);
      return(-1);
    }

    ptr = 0;
    for(j=0; j < nY*nDim; j++ ) {
      if(new_a[j] != 0) {
        sparse_A.index[nSel][ptr] = j;
        sparse_A.value[nSel][ptr++] = new_a[j];
      }
    }
  }
   
  new_col_H[nSel] = sq_norm_a;
  for(i=0; i < nSel; i++) {
    double tmp = 0;

    for(j=0; j < sparse_A.nz_dims[i]; j++) {
      tmp += new_a[sparse_A.index[i][j]]*sparse_A.value[i][j];
    }
      
    new_col_H[i] = tmp;
  }

  return 0;
}


/*----------------------------------------------------------------------
  sparse_compute_output( output ) does the follwing:

  output = W'*data_X;
  ----------------------------------------------------------------------*/
int msvm_sparse_compute_output( double *output, void* user_data )
{
  uint32_t i,y;

  for(i=0; i < nData; i++) 
  {
    for(y=0; y < nY; y++)
    {
      output[LIBOCAS_INDEX(y,i,nY)] = dp_sparse_col(&W[y*nDim], data_X, i);
    }
  }
  
  return 0;
}

/*----------------------------------------------------------------------------------
  full_add_new_cut( new_col_H, new_cut, cut_length, nSel ) does the following:

    new_a = sum(data_X(:,find(new_cut ~=0 )),2);
    new_col_H = [full_A(:,1:nSel)'*new_a ; new_a'*new_a];
    full_A(:,nSel+1) = new_a;

  ---------------------------------------------------------------------------------*/
int msvm_full_add_new_cut( double *new_col_H, uint32_t *new_cut, uint32_t nSel, void* user_data)
{
  double sq_norm_a, *ptr;
  uint32_t i, j, y, y2;

  ptr = mxGetPr(data_X);

  memset(new_a, 0, sizeof(double)*nDim*nY);

  for(i=0; i < nData; i++)
  {
    y = (uint32_t)(data_y[i]-1);
    y2 = (uint32_t)new_cut[i];
    if(y2 != y)
    {
      for(j=0; j < nDim; j++ ) 
      {
        new_a[LIBOCAS_INDEX(j,y,nDim)] += ptr[LIBOCAS_INDEX(j,i,nDim)];
        new_a[LIBOCAS_INDEX(j,y2,nDim)] -= ptr[LIBOCAS_INDEX(j,i,nDim)];
      }
    }
  }

  /* compute new_a'*new_a and insert new_a to the last column of full_A */
  sq_norm_a = 0;
  for(j=0; j < nDim*nY; j++ ) {
    sq_norm_a += new_a[j]*new_a[j];
    full_A[LIBOCAS_INDEX(j,nSel,nDim*nY)] = new_a[j];
  }

  new_col_H[nSel] = sq_norm_a;
  for(i=0; i < nSel; i++) {
    double tmp = 0;

    for(j=0; j < nDim*nY; j++ ) {
      tmp += new_a[j]*full_A[LIBOCAS_INDEX(j,i,nDim*nY)];
    }
    new_col_H[i] = tmp;
  }

  return 0;
}


/*----------------------------------------------------------------------
  full_compute_output( output ) does the follwing:

  output = data_X'*W;
  ----------------------------------------------------------------------*/
int msvm_full_compute_output( double *output, void* user_data )
{
  uint32_t i, j, y;
  double *ptr, tmp;

  ptr = mxGetPr( data_X );

  for(i=0; i < nData; i++) 
  { 
    for(y=0; y < nY; y++)
    {
      tmp = 0;

      for(j=0; j < nDim; j++ ) 
      {
        tmp += W[LIBOCAS_INDEX(j,y,nDim)]*ptr[LIBOCAS_INDEX(j,i,nDim)];
      }
      
      output[LIBOCAS_INDEX(y,i,nY)] = tmp;
    }
  }
  
  return 0;
}

/*----------------------------------------------------------------------------------
  sparse_add_new_cut( new_col_H, new_cut, cut_length, nSel ) does the following:

    new_a = sum(data_X(:,find(new_cut ~=0 )),2);
    new_col_H = [sparse_A(:,1:nSel)'*new_a ; new_a'*new_a];
    sparse_A(:,nSel+1) = new_a;

  ---------------------------------------------------------------------------------*/
int sparse_add_new_cut( double *new_col_H, 
                         uint32_t *new_cut, 
                         uint32_t cut_length, 
                         uint32_t nSel, 
                         void* user_data )
{
/*  double *new_a, */
  double sq_norm_a;
  uint32_t i, j, nz_dims, ptr;

  memset(new_a, 0, sizeof(double)*nDim);
  
  for(i=0; i < cut_length; i++) {
    add_sparse_col(new_a, data_X, new_cut[i]);

    A0[nSel] += X0*data_y[new_cut[i]];    
  }
 
  /* compute new_a'*new_a and count number of non-zero dimensions */
  nz_dims = 0; 
  sq_norm_a = A0[nSel]*A0[nSel];
  for(j=0; j < nDim; j++ ) {
    if(new_a[j] != 0) {
      nz_dims++;
      sq_norm_a += new_a[j]*new_a[j];
    }
  }

  /* sparsify new_a and insert it to the last column  of sparse_A */
  sparse_A.nz_dims[nSel] = nz_dims;
  if(nz_dims > 0) {
    sparse_A.index[nSel] = NULL;
    sparse_A.value[nSel] = NULL;
    sparse_A.index[nSel] = mxCalloc(nz_dims,sizeof(uint32_t));
    sparse_A.value[nSel] = mxCalloc(nz_dims,sizeof(double));
    if(sparse_A.index[nSel]==NULL || sparse_A.value[nSel]==NULL)
    {
/*      mexErrMsgTxt("Not enough memory for vector sparse_A.index[nSel], sparse_A.value[nSel].");*/
      mxFree(sparse_A.index[nSel]);
      mxFree(sparse_A.value[nSel]);
      return(-1);
    }

    ptr = 0;
    for(j=0; j < nDim; j++ ) {
      if(new_a[j] != 0) {
        sparse_A.index[nSel][ptr] = j;
        sparse_A.value[nSel][ptr++] = new_a[j];
      }
    }
  }
   
  new_col_H[nSel] = sq_norm_a;
  for(i=0; i < nSel; i++) {
    double tmp = A0[nSel]*A0[i];

    for(j=0; j < sparse_A.nz_dims[i]; j++) {
      tmp += new_a[sparse_A.index[i][j]]*sparse_A.value[i][j];
    }
      
    new_col_H[i] = tmp;
  }

/*  mxFree( new_a );*/

  return 0;
}


/*----------------------------------------------------------------------------------
  full_add_new_cut( new_col_H, new_cut, cut_length, nSel ) does the following:

    new_a = sum(data_X(:,find(new_cut ~=0 )),2);
    new_col_H = [full_A(:,1:nSel)'*new_a ; new_a'*new_a];
    full_A(:,nSel+1) = new_a;

  ---------------------------------------------------------------------------------*/
int full_add_new_cut( double *new_col_H, 
                       uint32_t *new_cut, 
                       uint32_t cut_length, 
                       uint32_t nSel,
                       void* user_data)
{
/*  double *new_a, */
  double sq_norm_a, *ptr;
  uint32_t i, j;

  ptr = mxGetPr(data_X);

  memset(new_a, 0, sizeof(double)*nDim);


  for(i=0; i < cut_length; i++) {
    for(j=0; j < nDim; j++ ) {
      new_a[j] += ptr[LIBOCAS_INDEX(j,new_cut[i],nDim)];
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
  sparse_compute_output( output ) does the follwing:

  output = data_X'*W;
  ----------------------------------------------------------------------*/
int sparse_compute_output( double *output, void* user_data )
{
  uint32_t i;

  for(i=0; i < nData; i++) { 
    output[i] = data_y[i]*X0*W0 + dp_sparse_col(W, data_X, i);
  }
  
  return 0;
}

/*----------------------------------------------------------------------
  full_compute_output( output ) does the follwing:

  output = data_X'*W;
  ----------------------------------------------------------------------*/
int full_compute_output( double *output, void* user_data )
{
  uint32_t i, j;
  double *ptr, tmp;

  ptr = mxGetPr( data_X );

  for(i=0; i < nData; i++) { 
    tmp = data_y[i]*X0*W0;

    for(j=0; j < nDim; j++ ) {
      tmp += W[j]*ptr[LIBOCAS_INDEX(j,i,nDim)];
    }
    output[i] = tmp;
  }
  
  return 0;
}