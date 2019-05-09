/*--------------------------------------------------------------
  C++ wrapper of
       LIMITED MEMORY BFGS METHOD FOR LARGE SCALE OPTIMIZATION
                          JORGE NOCEDAL

  The C++ was obtained by converting original Fortran code from
      http://www.ece.northwestern.edu/~nocedal/lbfgs.html
  by f2c plus some manual changes, e.g. dependency on F2C lib
  was removed.

  (C) 2014 V.Franc
----------------------------------------------------------------*/


#ifndef LBFGS_H__
#define LBFGS_H__

#include <stdio.h>
#include <math.h>
#include <string.h>

#define LBFGS_MP 6
#define LBFGS_LP 6
#define LBFGS_GTOL 0.9
#define LBFGS_STPMIN 1e-20
#define LBFGS_STPMAX 1e20

typedef enum LbfgsErrorCode
{
    kMaxIterAchived          = 1,
    kConverged               = 0,
    kLineSearchFailed        = -1,
    kNegativeHessianDiag     = -2,
    kImproperInputArguments  = -3,
    kMemoryAllocationProblem = -10,
    kEvalFunctionFailed      = -11
} LbfgsErrorCode;


class CLBFGS
{
    int     mp;
    int     lp;
    double  gtol;
    double  stpmin;
    double  stpmax;

    double *x_;
    double *w_;
    double *grad_;
    double *diag_;
    double fval_;

    double gnorm_;
    double xnorm_;

    LbfgsErrorCode exitflag_;

    // original function translated from Fortran
    int lbfgs_(const int n, const int m, double *x, double *f, double *g,
               int *diagco, double *diag, int *iprint,
               const double eps, const double xtol, double *w, int *iflag);

    double ddot_(int n, double *dx, int *incx, double *dy, int *incy);

    int daxpy_(int n, double *da, double *dx, int *incx, double *dy, int *incy);

    int mcsrch_(int n, double *x, double *f,
                double *g, double *s, double *stp, double *ftol,
                double xtol, int *maxfev, int *info, int *nfev,
                double *wa);

    int mcstep_(double *stx, double *fx, double *dx,
                double *sty, double *fy, double *dy, double *stp,
                double *fp, double *dp, int *brackt, double *stpmin,
                double *stpmax, int *info);

protected:

    // dimension of the parameter vector
    int    n_;

    // First order oracle of the optimized function.
    // This is the only stuff that must be defined in the derived class.
    virtual bool EvalFunction( double* x, double &fval, double* grad ) = 0;

    // Print function
    virtual void Print( char *msg ) { printf("%s", msg); }

public:

    int     n()            { return n_; }     // dimension of parameter vector
    double* grad()         { return grad_; }
    double  gnorm()        { return gnorm_; }
    double* x()            { return x_; }
    double  x(int i)       { return (i < n_) ? x_[i] : log(0); }
    double  xnorm()        { return xnorm_; }
    double  fval( void)    { return fval_; }
    int     exitflag(void) { return exitflag_; }

    CLBFGS() : x_(NULL), w_(NULL) ,grad_(NULL), diag_(NULL) {}

    ~CLBFGS() {
        if( x_ )   delete[] x_;
        if( grad_) delete[] grad_;
        if( diag_) delete[] diag_;
        if( w_)    delete[] w_;
    }

    /*
    n        is an INT variable that must be set by the user to the
             number of variables. It is not altered by the routine.
             Restriction: N>0.
    x0       Initial solution; if not give (x0=NULL) x0 is set to all zeros.
    max_iter Maximal number of iterations.
    eps      is a positive DOUBLE PRECISION variable that must be set by
             the user, and determines the accuracy with which the solution
             is to be found. The subroutine terminates when
                         ||G|| < EPS max(1,||X||),
              where ||.|| denotes the Euclidean norm.
     m       is an INT variable that must be set by the user to
             the number of corrections used in the BFGS update. It
             is not altered by the routine. Values of M less than 3 are
             not recommended; large values of M will result in excessive
             computing time. 3<= M <=7 is recommended. Restriction: M>0.
     xtol    is a  positive DOUBLE PRECISION variable that must be set by
             the user to an estimate of the machine precision (e.g.
             10**(-16) on a SUN station 3/60). The line search routine will
             terminate if the relative width of the interval of uncertainty
             is less than XTOL.

    */
    LbfgsErrorCode Solve( const int n, const double *x0=NULL, const long max_iter = 1000000,
               const double eps=1e-5, const int m=5, const double xtol=1e-16);
};


#endif
