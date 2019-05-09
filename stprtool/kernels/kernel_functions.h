
/*#define __KERNEL_DEBUG*/

#ifndef kernel_functions_h
#define kernel_functions_h

extern int kernel_init( const mxArray* _dataA, const mxArray* _dataB, const mxArray *name, const mxArray *args );
extern int kernel_destroy(void);
extern double kernel_eval(long i, long j);
extern long kernel_getM(void);
extern long kernel_getN(void);
long kernel_getCounter( void );

#endif
