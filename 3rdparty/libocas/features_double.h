/*-----------------------------------------------------------------------
 * features_double.h: Helper functions for the OCAS solver working with 
 *                   features in double precision.
 *-------------------------------------------------------------------- */

#ifndef _features_double_h
#define _features_double_h

#include <stdint.h>

/* dense double features */
extern int full_compute_output( double *output, void* user_data );
extern int full_add_new_cut( double *new_col_H, 
                       uint32_t *new_cut, 
                       uint32_t cut_length, 
                       uint32_t nSel,
                       void* user_data);

/* sparse double features */
extern int sparse_add_new_cut( double *new_col_H, 
                         uint32_t *new_cut, 
                         uint32_t cut_length, 
                         uint32_t nSel, 
                         void* user_data );
extern int sparse_compute_output( double *output, void* user_data );


/* dense double features for multi-class solver */
extern int msvm_full_add_new_cut( double *new_col_H, uint32_t *new_cut, uint32_t nSel, void* user_data);
extern int msvm_full_compute_output( double *output, void* user_data );

/* sparse double features for multi-class solver */
extern int msvm_sparse_add_new_cut( double *new_col_H, uint32_t *new_cut, uint32_t nSel, void* user_data );
extern int msvm_sparse_compute_output( double *output, void* user_data );


#endif
