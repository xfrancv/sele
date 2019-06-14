/*-----------------------------------------------------------------------
 * ocas_helper.h: Implementation of helper functions for the OCAS solver.
 *
 * It supports both sparse and dense matrices and loading data from
 * the SVM^light format.
 *-------------------------------------------------------------------- */

#ifndef _svmocas_bool_helper_h
#define _svmocas_bool_helper_h


int full_bool_compute_output( double *output, void* user_data );
int full_bool_add_new_cut( double *new_col_H, 
                       uint32_t *new_cut, 
                       uint32_t cut_length, 
                       uint32_t nSel,
                       void* user_data);


#endif
