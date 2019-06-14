/*-----------------------------------------------------------------------
 * features_int8.h: Helper functions for the OCAS solver working with 
 *                 INT8 features.
 *
 *-------------------------------------------------------------------- */

#ifndef _features_int8_h
#define _features_int8_h

#include <stdint.h>


extern int full_int8_compute_output( double *output, void* user_data );
extern int full_int8_add_new_cut( double *new_col_H, 
                       uint32_t *new_cut, 
                       uint32_t cut_length, 
                       uint32_t nSel,
                       void* user_data);

#endif
