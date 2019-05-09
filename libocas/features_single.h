/*-----------------------------------------------------------------------
 * features_single.h: Helper functions for the OCAS solver working with 
 *                   features in double precision.
 *-------------------------------------------------------------------- */

#ifndef _features_single_h
#define _features_single_h

#include <stdint.h>

/* dense double features */
extern int full_single_compute_output( double *output, void* user_data );
extern int full_single_add_new_cut( double *new_col_H, 
                       uint32_t *new_cut, 
                       uint32_t cut_length, 
                       uint32_t nSel,
                       void* user_data);


#endif
