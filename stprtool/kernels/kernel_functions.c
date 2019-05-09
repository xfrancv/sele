/* --------------------------------------------------------------------
 Library for computing kernel functions.

 Implemented kernels:
    linear 
    poly
    rbf
    sigmoidal


 Written (W) 1999-2010, Written by Vojtech Franc
 Copyright (C) 1999-2010, Czech Technical University in Prague
-------------------------------------------------------------------- */

#include <math.h>
#include <string.h>
#include "mex.h"
#include "matrix.h"
#include "kernel_functions.h"

#define MAX(A,B)   (((A) > (B)) ? (A) : (B) )
#define MIN(A,B)   (((A) < (B)) ? (A) : (B) )


#define DATA_TYPE_UNKNOWN -1
#define DATA_TYPE_DENSE 0

#define KERNEL_NAME_LINEAR      0
#define KERNEL_NAME_RBF         1
#define KERNEL_NAME_POLY        2
#define KERNEL_NAME_SIGMOID     3
#define KERNEL_NAME_PRECOMPUTED 4

#define KERNEL_TYPE_UNKNOWN          -1
#define KERNEL_TYPE_LINEAR_DENSE      0
#define KERNEL_TYPE_RBF_DENSE         1
#define KERNEL_TYPE_POLY_DENSE        2
#define KERNEL_TYPE_SIGMOID_DENSE     3
#define KERNEL_TYPE_PRECOMPUTED_DENSE 4

/****************************************************/
/**** Global variables ******************************/
/****************************************************/

static char *SUPPORTED_KERNELS[] = {"linear","rbf","poly","sigmoid","precomputed"};

static long nDim;           /* dimension of vectorial data */
static int  kernel_type;    /* 0 - linear_dense        id=0 name="linear" and data is dense double matrix
                               1 - rbf_dense           id=1 name="rbf" and data is dense double matrix
                               2 - poly_dense          id=2 name="poly" and data is dense double matrix
                               3 - sigmoid_dense       id=3 name="sigmoid" and data is dense double matrix
                               4 - precomputed_dense   id=4 name="precomputed" and data is dense double matrix
                            */

static int data_type;       /* 0 - dense double matrices
                               1 - sparse double matrices
                             */


static long kernel_cnt;     /* counter of kernel evaluations */

static long nDims;          /* dimensionality of vectorial data */
static long M;              /* number of objects in data A */
static long N;              /* number of objects in data B */
static mxArray *dataA = 0;
static mxArray *dataB = 0;
static double *dense_matA = 0;
static double *dense_matB = 0;

static double *norm_sq_A = 0;
static double *norm_sq_B = 0;

static double rbf_gamma;

static double poly_arg0 = 0;
static double poly_arg1 = 1;
static double poly_degree;

static double sigmoid_arg0;
static double sigmoid_arg1;

/*************************************************************************************
  Helper functions
 *************************************************************************************/

/* -------------------------------------------------------------------
 Compute dot product between a-th column of dense_matA and
 b-th column of dense_matB.
------------------------------------------------------------------- */
double dense_dot_prod( long a, long b)
{
   double dp = 0;
   long i;

   for( i = 0; i < nDims; i++ ) 
      dp += *(dense_matA+(a*nDims)+i) * *(dense_matB+(b*nDims)+i);

   return( dp );
}

double dense_norm_sq_A( long a )
{
   double dp = 0;
   long i;

   for( i = 0; i < nDims; i++ ) 
     dp += *(dense_matA+(a*nDims)+i) * *(dense_matA+(a*nDims)+i);

   return( dp );
}

double dense_norm_sq_B( long a )
{
   double dp = 0;
   long i;

   for( i = 0; i < nDims; i++ ) 
     dp += *(dense_matB+(a*nDims)+i) * *(dense_matB+(a*nDims)+i);

   return( dp );
}

/**************************************************************/
/* LINEAR KERNEL ON DENSE DATA                                */
/**************************************************************/

/* double* linear_dense_get_col(long i) */

/**************************************************************/
/* RBF KERNEL ON DENSE DATA                                   */
/**************************************************************/
int rbf_init(const mxArray *args)
{
  long i;

  if( !mxIsDouble(args) || mxGetM(args) != 1 || mxGetN(args) != 1)
  {
    mexPrintf("Improper arguments for RBF kernel.\n");
    return(0);
  }
  rbf_gamma = mxGetScalar(args);

  norm_sq_A = mxCalloc( M, sizeof( double ));
  if(!norm_sq_A)
  {
    mexPrintf("Not enough memory to allocate norm_sq_A.\n");
    return(0);
  }

  if(data_type == DATA_TYPE_DENSE)
  {
    for(i=0; i < M; i++)
      norm_sq_A[i] = dense_norm_sq_A( i );
  }

  if( dense_matA == dense_matB)
  {
    norm_sq_B = norm_sq_A;
  }
  else
  {
    norm_sq_B = mxCalloc( N, sizeof( double ));
    if(!norm_sq_B)
    {
      mxFree( norm_sq_A );
      mexPrintf("Not enough memory to allocate norm_sq_B.\n");
      return(0);
    }

    if(data_type == DATA_TYPE_DENSE)
    {
      for(i=0; i < N; i++)
      norm_sq_B[i] = dense_norm_sq_B( i );    
    }
  }

  return(1);
}

int rbf_destroy(void)
{
  if( dense_matA == dense_matB )
  {
    mxFree( norm_sq_A );
  }
  else
  {
    mxFree( norm_sq_A );
    mxFree( norm_sq_B );
  }
  return(1);
}

/**************************************************************/
/* POLYNOMIAL KERNEL ON DENSE DATA                            */
/**************************************************************/

int poly_dense_init(const mxArray *args )
{
  int max = MAX(mxGetM(args),mxGetN(args));
  int min = MIN(mxGetM(args),mxGetN(args));
  double *tmp;

  if( !mxIsDouble(args) || max < 1 || max > 3 || min > 1)
  {
    mexPrintf("Improper arguments for polynomial kernel.\n");
    return(0);
  }

  tmp = mxGetPr(args);
  poly_degree = tmp[0];
  if(max > 1)
    poly_arg0 = tmp[1];
  if(max > 2)
    poly_arg1 = tmp[2];

#ifdef __KERNEL_DEBUG
  mexPrintf("poly: degree=%f, arg0=%f, arg1=%f\n", poly_degree, poly_arg0, poly_arg1);
#endif
  
  return(1);
}

/**************************************************************/
/* SIGMOID KERNEL ON DENSE DATA                               */
/**************************************************************/

int sigmoid_dense_init(const mxArray *args )
{
  int max = MAX(mxGetM(args),mxGetN(args));
  int min = MIN(mxGetM(args),mxGetN(args));
  double *tmp;

  if( !mxIsDouble(args) || max != 2 || min > 1 )
  {
    mexPrintf("Improper arguments for sigmoid kernel.\n");
    return(0);
  }

  tmp = mxGetPr(args);
  sigmoid_arg0 = tmp[0];
  sigmoid_arg1 = tmp[1];

#ifdef __KERNEL_DEBUG
  mexPrintf("sigmoid: arg0=%f, arg1=%f\n", sigmoid_arg0, sigmoid_arg1);
#endif
  
  return(1);
}

/**************************************************************/
/* PRECOMPUTED KERNEL                                         */
/**************************************************************/
int precomputed_init(void)
{
  if( dense_matA != dense_matB || nDims != N )
  {
    mexPrintf("Improper input data for precomputed kernel.\n");
    return(0);
  }
  return(1);
}

/*************************************************************************************
  Initialization function. 
  It extracts anch checks the inputs and precomputes/allocates what is ncessary. 
 *************************************************************************************/
int kernel_init( const mxArray* _dataA, const mxArray* _dataB, const mxArray *name, const mxArray *args )
{
  int nKernels;
  int name_len;
  int exitflag = 0;
  int name_id;
  int i;
  char *name_str = NULL;

  kernel_cnt = 0;
    
  /****************************************************/
  /* Convert kernel name (string) to its numerical id */
  /****************************************************/
  if( mxIsChar( name ) != 1) 
  {
    mexPrintf("Kernel identifier must be string.\n");
    goto clean_up;
  }

  name_len  = (mxGetM(name) * mxGetN(name)) + 1;
  name_str = mxCalloc( name_len, sizeof( char ));
  if( !name_str) 
  {
    mexPrintf("Cannot allocate memory for kernel identifier string.\n");
    goto clean_up;
  }

  mxGetString( name, name_str, name_len );  
  nKernels = sizeof( SUPPORTED_KERNELS )/sizeof( char * );

  name_id = -1;
  for( i = 0; i < nKernels; i++ ) 
  {
    if( strcmp( name_str, SUPPORTED_KERNELS[i] )==0 ) 
    {
      name_id = i;
      break;
    }
  }

  if(name_id == -1)
  {
    mexPrintf("Unknown kernel identifier.\n");
    goto clean_up;
  }

  /********************/
  /* Check data types */
  /********************/
  data_type = DATA_TYPE_UNKNOWN;
  if( mxIsDouble(_dataA) && mxIsDouble(_dataB) && /* double */
      !mxIsEmpty(_dataA) && !mxIsEmpty(_dataB) && /* non-empty */
      mxGetM(_dataA) == mxGetM(_dataB))           /* dimesion must equal */
  {
    /* DENSE MATRICES IN DOUBLE PRECISION */
    data_type = DATA_TYPE_DENSE; 

    dense_matA = mxGetPr(_dataA);
    dense_matB = mxGetPr(_dataB);

    nDims = mxGetM(_dataA);
    M = mxGetN(_dataA);
    N = mxGetN(_dataB);
  }

  if(data_type == DATA_TYPE_UNKNOWN)
  {
    mexPrintf("Input data are of the type which is not supported.\n");
    goto clean_up;
  }

  /**********************/
  /* Select kernel type */
  /**********************/
  kernel_type = KERNEL_TYPE_UNKNOWN;
  if( name_id == KERNEL_NAME_LINEAR && data_type == DATA_TYPE_DENSE)
    kernel_type = KERNEL_TYPE_LINEAR_DENSE; /* linear kernel on dense matrices */

  if( name_id == KERNEL_NAME_RBF && data_type == DATA_TYPE_DENSE)
    kernel_type = KERNEL_TYPE_RBF_DENSE; /* rbf kernel on dense matrices */

  if( name_id == KERNEL_NAME_POLY && data_type == DATA_TYPE_DENSE)
    kernel_type = KERNEL_TYPE_POLY_DENSE; /* polynomial kernel on dense matrices */

  if( name_id == KERNEL_NAME_SIGMOID && data_type == DATA_TYPE_DENSE)
    kernel_type = KERNEL_TYPE_SIGMOID_DENSE; /* sigmoid kernel on dense matrices */

  if( name_id == KERNEL_NAME_PRECOMPUTED && data_type == DATA_TYPE_DENSE)
    kernel_type = KERNEL_TYPE_PRECOMPUTED_DENSE; /* precomputed kernel */

#ifdef __KERNEL_DEBUG
  mexPrintf("kenel: name=%s, id=%d, type=%d\n", name_str, name_id, kernel_type);
  mexPrintf("M=%d, N=%d, nDims=%d\n", M, N, nDims);
#endif

  /***********************/
  /* Call initialization */
  /***********************/
  switch( kernel_type )
  {
    case KERNEL_TYPE_LINEAR_DENSE:   /* linear kernel on dense matrices */
      exitflag = 1;
      break;

    case KERNEL_TYPE_RBF_DENSE:   /* rbf kernel on dense matrices */
      exitflag = rbf_init(args);
      break;

    case KERNEL_TYPE_POLY_DENSE:   /* polynomial kernel on dense matrices */
      exitflag = poly_dense_init(args);
      break;

    case KERNEL_TYPE_SIGMOID_DENSE:   /* sigmoid kernel on dense matrices */
      exitflag = sigmoid_dense_init(args);
      break;

    case KERNEL_TYPE_PRECOMPUTED_DENSE:   /* precomputed kernel */
      exitflag = precomputed_init();
      break;

    default:
      mexPrintf("Kernel identifier and input data do not match.\n");
      goto clean_up;
  }

 clean_up:
  mxFree(name_str);

  return(exitflag);
}


/*************************************************************************************
  Destroy function.
  Free temporary variables if used. 
*************************************************************************************/
int kernel_destroy(void)
{
  int exitflag = 0;

  switch(kernel_type)
  {
     case KERNEL_TYPE_LINEAR_DENSE:   
       exitflag = 1;
       break;

     case KERNEL_TYPE_RBF_DENSE:   
       exitflag = rbf_destroy();
       break;

     case KERNEL_TYPE_POLY_DENSE:   
       exitflag = 1;
       break;

     case KERNEL_TYPE_SIGMOID_DENSE:   
       exitflag = 1;
       break;

     case KERNEL_TYPE_PRECOMPUTED_DENSE: 
       exitflag = 1;
       break;
  }

  return(exitflag);
}

/*************************************************************************************
  Function returning single entry of the kernel matrix.
*************************************************************************************/
double kernel_eval(long i, long j)
{
#ifdef __KERNEL_DEBUG
  if(i < 0 || i >= M)
    mexErrMsgTxt("kernel_eval: Row index out of range.");

  if(j < 0 || j >= N)
    mexErrMsgTxt("kernel_eval: Column index out of range.");
#endif

  kernel_cnt++;

  switch( kernel_type )
  {
    case KERNEL_TYPE_LINEAR_DENSE: 
      return( dense_dot_prod( i, j ) );

    case KERNEL_TYPE_RBF_DENSE: 
      return( exp( -rbf_gamma * ( norm_sq_A[i] + norm_sq_B[j] - 2*dense_dot_prod( i, j ) ) ) );

    case KERNEL_TYPE_POLY_DENSE:   
      return( pow( poly_arg1 * dense_dot_prod( i, j ) + poly_arg0 , poly_degree ) );

    case KERNEL_TYPE_SIGMOID_DENSE:   
      return( tanh( sigmoid_arg1 * dense_dot_prod( i, j ) + sigmoid_arg0 ) );

    case KERNEL_TYPE_PRECOMPUTED_DENSE: 
      return( dense_matA[i + j*nDims] );    
  }
}


/*************************************************************************************
  Returns number of objects in dataA.
*************************************************************************************/
long kernel_getM( void )
{
  return( M );
}

/*************************************************************************************
  Returns number of objects in dataB.
*************************************************************************************/
long kernel_getN( void )
{
  return( N );
}

/*************************************************************************************
  Returns number of objects in dataB.
*************************************************************************************/
long kernel_getCounter( void )
{
  return( kernel_cnt );
}
