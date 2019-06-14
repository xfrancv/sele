/*-----------------------------------------------------------------------
 * ocas_helper.h: Implementation of helper functions for the OCAS solver.
 *
 *-------------------------------------------------------------------- */

#ifndef _ocas_helper_h
#define _ocas_helper_h

#include <stdint.h>
#include "libocas.h"

#ifdef LIBOCAS_MATLAB

#include <mex.h>

#if !defined(MX_API_VER) || MX_API_VER<0x07040000
#define mwSize int
#define INDEX_TYPE_T int
#define mwIndex int
#else
#define INDEX_TYPE_T mwSize
#endif

#else

#define mwSize int
#define mwIndex int

#include "sparse_mat.h"

#endif

typedef struct {
  double **value;
  uint32_t **index;
  uint32_t *nz_dims;    
} cutting_plane_buf_T;


extern mxArray *data_X;
extern uint32_t nDim, nData, nY;
extern double *data_y;
extern cutting_plane_buf_T sparse_A;
extern double *full_A;
extern double *W;
extern double *oldW;
extern double *new_a;

extern double *A0;
extern double W0;
extern double oldW0;
extern double X0;


/** helper functions for OCAS solver **/
extern void ocas_print(ocas_return_value_T value);
extern void ocas_print_null(ocas_return_value_T value);
extern int qsort_data(double* value, double* data, uint32_t size);

/** helper functions for two-class SVM  solver **/
extern double update_W( double t, void* user_data );
/* dense cutting plane buffer*/
extern void full_compute_W( double *sq_norm_W, double *dp_WoldW, double *alpha, uint32_t nSel, void* user_data );
/* sparse cutting plane bugger */
extern void sparse_compute_W( double *sq_norm_W, double *dp_WoldW, double *alpha, uint32_t nSel, void* user_data );

/** helper function for two-class SVM solver with additional positivity contraints on W **/
int full_add_nnw_constr(uint32_t idx, uint32_t nSel, void* user_data);
void clip_neg_W( uint32_t num_pw_constr, uint32_t *pw_idx, void* user_data );
int sparse_add_nnw_constr(uint32_t idx, uint32_t nSel, void* user_data);


/** helper functions for multi-class SVM  solver **/
extern double msvm_update_W( double t, void* user_data );
/* dense cutting plane buffer*/
extern void msvm_full_compute_W( double *sq_norm_W, double *dp_WoldW, double *alpha, uint32_t nSel, void* user_data );
/* sparse cutting plane bugger */
extern void msvm_sparse_compute_W( double *sq_norm_W, double *dp_WoldW, double *alpha, uint32_t nSel, void* user_data );


/** parallelized implementation of helper functions for two-class OCAS solver **/
extern void destroy_parallel_ocas(void);
extern int init_parallel_ocas(int number_of_threads);
extern int parallel_sparse_compute_output( double *output, void* user_data );
extern int parallel_qsort_data(double* value, double* data, uint32_t size);
extern int parallel_sparse_add_new_cut( double *new_col_H,uint32_t *new_cut, uint32_t cut_length,uint32_t nSel,void* user_data );


/** functions for working with sparse vectors **/
extern void mul_sparse_col(double alpha, mxArray *sparse_mat, uint32_t col);
extern void add_sparse_col(double *full_vec, mxArray *sparse_mat, uint32_t col);
extern void subtract_sparse_col(double *full_vec, mxArray *sparse_mat, uint32_t col);
extern double dp_sparse_col(double *full_vec, mxArray *sparse_mat, uint32_t col);


/** auxcilirary functions **/
extern int load_svmlight_file(char *fname, int verb);
extern double compute_auc(double *score, int *label, uint32_t nData);
extern int load_regconsts(char *fname, double **vec_C, uint32_t *len_vec_C, int verb);
extern double get_time(void);


#endif
