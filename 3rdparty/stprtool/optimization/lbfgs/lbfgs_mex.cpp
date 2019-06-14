/*=================================================================
  MEX interface to LBFGS solver.

  VF
 *=================================================================*/

#include <mex.h>

#include "lbfgs.h"


class Solver : public CLBFGS {

  mxArray *lhs_[2];
  mxArray *rhs_[2];  
  char    *fun_;
  int     dim_;
  int     verb_;

public:
  Solver( int n, char* fun, int verb, const mxArray *params ) : CLBFGS () 
  {
    verb_ = verb;
    dim_  = n;

    rhs_[0] = (mxArray*)params;

    rhs_[1] = mxCreateDoubleMatrix(n, 1, mxREAL );
    if( !rhs_[1] )
      mexErrMsgTxt("Memory allocation problem.\n");

    int buflen = strlen( fun );
    fun_ =  (char*)mxMalloc(buflen);
    strcpy( fun_, fun );
  }

  virtual void Print( char *msg ) { 
    if( verb_ ) mexPrintf("%s", msg); 
  }


  // call 1st order oracle at point x
  bool EvalFunction( double* x, double &fval, double* grad )
  {
    double* ptr = mxGetPr( rhs_[1] );

    memcpy( ptr, x, sizeof(double)*dim_);
    
    int ef = mexCallMATLAB( 2, lhs_, 2 , rhs_, fun_);
    if( ef ) {
      mexPrintf("Calling %s failed.", fun_);
      mexErrMsgTxt("\n");
    }

    fval = mxGetScalar(lhs_[0]);
           
    ptr = mxGetPr( lhs_[1] );
    memcpy( grad, ptr, sizeof(double)*dim_ );
      
    mxDestroyArray( lhs_[0]);
    mxDestroyArray( lhs_[1]);

    return true;
  }
};


void mexFunction( int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[] )
{
    if( nrhs < 3 )
        mexErrMsgTxt("LBFGS limited memory BFGS of J.Nocedal.\n\n"
                 "Synopsis:\n"
                 "  [x,fval,exitFlag] = lbfgs(Params,fun, x0 )\n"
                 "  [x,fval,exitFlag] = lbfgs(Params,fun, x0, Opt )\n"
                 "Input:\n"
                 "  Params [] arbitrary variable that is past to fun()\n"     
                 "  fun [string] name of 1st order oracle function called as\n"
                 "         [fval,grad] = fun( Params, x )\n"
                 "  x0 [N x 1] initial solution.\n"
                 "  Opt [struct] optimizer settings:\n"
                 "   .maxIter [1x1] maximal number of iterations (def 10000000).\n"
                 "   .eps     [1x1] accuracy of the solution (def 1e-5).\n"
                 "                  The algorithm terminates if ||grad|| < eps max(1,||x||)\n"
                 "   .m       [1x1] the number of corrections used in the BFGS update. \n"
                 "   .xtol    [1x1] an estimate of the machine precision (def 1e-16)\n"
                 "   .verb    [1x1] if then display progress info (def 0).\n"
                 "Output:\n"
                 "   x        [Nx1] solution vector. \n"
                 "   fval     [Nx1] function value.\n"
                 "   exitflag [Nx1] 0   .. converged.\n"
                 "                  1   .. maximal number of iterations achieved.\n"
                 "                  < 0 .. failure.\n"
                 "\n");

    /* arg 2: fun */
    if( !mxIsChar(prhs[1]) )
      mexErrMsgTxt("Second argument must be a string.");
    
    char *fun;
    int  buflen;
    int  status;
    buflen = mxGetN(prhs[1])*sizeof(mxChar)+1;
    fun    = (char*)mxMalloc(buflen);
    status = mxGetString(prhs[1], fun, buflen);
    

    /* arg 3: x0*/
    double *x0;
    int    n;
    if( mxIsEmpty( prhs[2]))
    {
      // call [risk,subgrad] = fun(Params), without the second argumnet,
      // to get the number of variables from subgrad
      mxArray *lhs[2];
      mxArray *rhs[1];

      rhs[0] = (mxArray*)prhs[0];

      int ef = mexCallMATLAB( 2, lhs, 1 , rhs, fun );
      if( ef ) 
      {
        mexPrintf("Calling %s failed.", fun);
        mexErrMsgTxt("\n");
      }      
      n  = mxGetM( lhs[1] );
      x0 = (double*)mxCalloc(n,sizeof(double));

    }
    else
    {
      mwSize num_rows = mxGetM( prhs[2] );
      mwSize num_cols = mxGetN( prhs[2] );
      if( (num_rows > 1 && num_cols > 1 ) || mxIsComplex(prhs[2]) || !mxIsDouble( prhs[2] ) )
          mexErrMsgTxt("Third argument must be a double vector.");

      x0  = mxGetPr( prhs[2] );
      n   = (num_rows >= num_cols) ? num_rows : num_cols;
    }

    /* arg 4: Opt */
    long   max_iter = 1000000;
    double eps = 1e-5;
    double xtol = 1e-16;
    int    m = 5;
    int    verb = 0;

    if( nrhs > 3 )
    {
 
      if( !mxIsStruct( prhs[3] ) )
        mexErrMsgTxt("Third argument must be a structure.");

      mxArray *ptr;
      ptr = mxGetField( prhs[3], 0, "maxIter");
      if( ptr != NULL ) max_iter = (long)mxGetScalar( ptr );

      ptr = mxGetField( prhs[3], 0, "eps");
      if( ptr != NULL ) eps = mxGetScalar( ptr );

      ptr = mxGetField( prhs[3], 0, "xtol");
      if( ptr != NULL ) xtol = mxGetScalar( ptr );

      ptr = mxGetField( prhs[3], 0, "m");
      if( ptr != NULL ) m = (int)mxGetScalar( ptr );

      ptr = mxGetField( prhs[3], 0, "verb");
      if( ptr != NULL ) verb = (int)mxGetScalar( ptr );
    }

    ////////////////////////
    if( verb ) {
      mexPrintf("Setting:\n");
      mexPrintf("fun    : %s\n", fun );
      mexPrintf("n      : %d\n", n);
      mexPrintf("maxIter: %ld\n", max_iter );
      mexPrintf("eps    : %f\n", eps );
      mexPrintf("m      : %d\n", m );
      mexPrintf("xtol   : %d\n", xtol );
    }
    

    ////////////////////

    Solver solver( n, fun, verb, prhs[0] );

    solver.Solve( n, x0, max_iter, eps, m, xtol );

    ////////////////////

    // output arg 1
    if( nlhs >= 1 ) {
      plhs[0] = mxCreateDoubleMatrix(n,1,mxREAL);
      double *pr = mxGetPr( plhs[0] );
      memcpy( pr, solver.x(), n*sizeof( double ) );      
    }

    // output arg 2
    if( nlhs >= 2 ) {
      plhs[1] = mxCreateDoubleScalar( solver.fval() );
    }

    // output arg 3
    if( nlhs >= 3 ) {
      plhs[2] = mxCreateDoubleScalar( (double)solver.exitflag() );
    }

    return;
}
