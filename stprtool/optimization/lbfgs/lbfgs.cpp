/* lbfgs.f -- translated by f2c (version 20090411).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include "lbfgs.h"


#define LBFGS_TRUE (1)
#define LBFGS_FALSE (0)

#define LBFGS_ABS(x) ((x) >= 0 ? (x) : -(x))
#define LBFGS_MIN(a,b) ((a) <= (b) ? (a) : (b))
#define LBFGS_MAX(a,b) ((a) >= (b) ? (a) : (b))


LbfgsErrorCode CLBFGS::Solve( const int n, const double *x0, const long max_iter,
           const double eps, const int m, const double xtol)
{
    char msg[100];

    // constants used by LBFGS
    mp     = LBFGS_MP;
    lp     = LBFGS_LP;
    gtol   = LBFGS_GTOL;
    stpmin = LBFGS_STPMIN;
    stpmax = LBFGS_STPMAX;

    // user defined parameters
    n_        = n;

    // reinit working variables
    if( x_ ) delete[] x_;
    if( grad_) delete[] grad_;
    if( diag_) delete[] diag_;
    if( w_) delete[] w_;

    x_    = NULL;
    grad_ = NULL;
    diag_ = NULL;
    w_    = NULL;

    x_    = new double[n];
    grad_ = new double[n];
    diag_ = new double[n];
    w_    = new double[ n*(2*m+1) + 2*m];

    if( !x_ || !grad_ || !diag_ || !w_ )
    {
        sprintf(msg, "Memory allocation error.\n");
        Print(msg);
        exitflag_ = kMemoryAllocationProblem;
        return exitflag_;
    }

    // initial solution
    if( x0 == NULL)
        memset( x_, 0, n*sizeof(double) );
    else
        memcpy( x_, x0, n*sizeof(double) );

    //
    int diagco = LBFGS_FALSE;
    int iprint[2];
    iprint[0] = 1;
    iprint[1] = 0;
    int iflag;

    for( long i = 1; i <= max_iter; i++ )
    {
        if( !EvalFunction( x_, fval_, grad_ ) )
        {
            sprintf(msg, "Calling EvalFunction() failed.\n" );
            Print(msg);
            return kEvalFunctionFailed;
        }

        lbfgs_(n_, m, x_, &fval_, grad_, &diagco, diag_, iprint, eps, xtol, w_, &iflag);

        sprintf(msg, "iter=%6ld    fval=%f\n", i, fval_);
        Print(msg);

        if( iflag <= 0 )
        {
            exitflag_= (LbfgsErrorCode)iflag;
            switch( exitflag_ )
            {
                case kMaxIterAchived:
                    sprintf(msg,"Maximal number of iterations achieved.\n");
                    break;
                case kConverged:
                    sprintf(msg,"Converged.\n");
                    break;
                case kLineSearchFailed:
                    sprintf(msg,"Line search failed.\n");
                    break;
                case kNegativeHessianDiag:
                    sprintf(msg,"Diagonal element of Hessian iverse negative.\n");
                    break;
                case kImproperInputArguments:
                    sprintf(msg,"Improper input arguments.\n");
                    break;
            }
            Print(msg);

            return exitflag_;
        }
    }

    exitflag_ = kMaxIterAchived;
    return exitflag_;
 }



/*     ******************************************************************************/
/**/
/*        LIMITED MEMORY BFGS METHOD FOR LARGE SCALE OPTIMIZATION */
/*                          JORGE NOCEDAL */
/*                        *** July 1990 *** */
/*     This subroutine solves the unconstrained minimization problem */
/**/
/*                      min F(x),    x= (x1,x2,...,xN), */
/**/
/*      using the limited memory BFGS method. The routine is especially */
/*      effective on problems involving a large number of variables. In */
/*      a typical iteration of this method an approximation Hk to the */
/*      inverse of the Hessian is obtained by applying M BFGS updates to */
/*      a diagonal matrix Hk0, using information from the previous M steps. */
/*      The user specifies the number M, which determines the amount of */
/*      storage required by the routine. The user may also provide the */
/*      diagonal matrices Hk0 if not satisfied with the default choice. */
/*      The algorithm is described in "On the limited memory BFGS method */
/*      for large scale optimization", by D. Liu and J. Nocedal, */
/*      Mathematical Programming B 45 (1989) 503-528. */
/**/
/*      The user is required to calculate the function value F and its */
/*      gradient G. In order to allow the user complete control over */
/*      these computations, reverse  communication is used. The routine */
/*      must be called repeatedly under the control of the parameter */
/*      IFLAG. */
/**/
/*      The steplength is determined at each iteration by means of the */
/*      line search routine MCVSRCH, which is a slight modification of */
/*      the routine CSRCH written by More' and Thuente. */
/**/
/*      The calling statement is */
/*          CALL LBFGS(N,M,X,F,G,DIAGCO,DIAG,IPRINT,EPS,XTOL,W,IFLAG) */
/*      where */
/*     N       is an INT variable that must be set by the user to the */
/*             number of variables. It is not altered by the routine. */
/*             Restriction: N>0. */
/*     M       is an INT variable that must be set by the user to */
/*             the number of corrections used in the BFGS update. It */
/*             is not altered by the routine. Values of M less than 3 are */
/*             not recommended; large values of M will result in excessive */
/*             computing time. 3<= M <=7 is recommended. Restriction: M>0. */
/*     X       is a DOUBLE PRECISION array of length N. On initial entry */
/*             it must be set by the user to the values of the initial */
/*             estimate of the solution vector. On exit with IFLAG=0, it */
/*             contains the values of the variables at the best point */
/*             found (usually a solution). */
/*     F       is a DOUBLE PRECISION variable. Before initial entry and on */
/*             a re-entry with IFLAG=1, it must be set by the user to */
/*             contain the value of the function F at the point X. */
/*     G       is a DOUBLE PRECISION array of length N. Before initial */
/*             entry and on a re-entry with IFLAG=1, it must be set by */
/*             the user to contain the components of the gradient G at */
/*             the point X. */
/*     DIAGCO  is a INT variable that must be set to .TRUE. if the */
/*             user  wishes to provide the diagonal matrix Hk0 at each */
/*             iteration. Otherwise it should be set to .FALSE., in which */
/*             case  LBFGS will use a default value described below. If */
/*             DIAGCO is set to .TRUE. the routine will return at each */
/*             iteration of the algorithm with IFLAG=2, and the diagonal */
/*              matrix Hk0  must be provided in the array DIAG. */
/**/
/**/
/*     DIAG    is a DOUBLE PRECISION array of length N. If DIAGCO=.TRUE., */
/*             then on initial entry or on re-entry with IFLAG=2, DIAG */
/*             it must be set by the user to contain the values of the */
/*             diagonal matrix Hk0.  Restriction: all elements of DIAG */
/*             must be positive. */
/**/
/*     IPRINT  is an INT array of length two which must be set by the */
/*             user. */
/**/
/*             IPRINT(1) specifies the frequency of the output: */
/*                IPRINT(1) < 0 : no output is generated, */
/*                IPRINT(1) = 0 : output only at first and last iteration, */
/*                IPRINT(1) > 0 : output every IPRINT(1) iterations. */
/**/
/*             IPRINT(2) specifies the type of output generated: */
/*                IPRINT(2) = 0 : iteration count, number of function */
/*                                evaluations, function value, norm of the */
/*                                gradient, and steplength, */
/*                IPRINT(2) = 1 : same as IPRINT(2)=0, plus vector of */
/*                                variables and  gradient vector at the */
/*                                initial point, */
/*                IPRINT(2) = 2 : same as IPRINT(2)=1, plus vector of */
/*                                variables, */
/*                IPRINT(2) = 3 : same as IPRINT(2)=2, plus gradient vector. */
/**/
/*     EPS     is a positive DOUBLE PRECISION variable that must be set by */
/*             the user, and determines the accuracy with which the solution */
/*             is to be found. The subroutine terminates when */
/**/
/*                         ||G|| < EPS max(1,||X||), */
/**/
/*             where ||.|| denotes the Euclidean norm. */
/**/
/*     XTOL    is a  positive DOUBLE PRECISION variable that must be set by */
/*             the user to an estimate of the machine precision (e.g. */
/*             10**(-16) on a SUN station 3/60). The line search routine will */
/*             terminate if the relative width of the interval of uncertainty */
/*             is less than XTOL. */
/**/
/*     W       is a DOUBLE PRECISION array of length N(2M+1)+2M used as */
/*             workspace for LBFGS. This array must not be altered by the */
/*             user. */
/**/
/*     IFLAG   is an INT variable that must be set to 0 on initial entry */
/*             to the subroutine. A return with IFLAG<0 indicates an error, */
/*             and IFLAG=0 indicates that the routine has terminated without */
/*             detecting errors. On a return with IFLAG=1, the user must */
/*             evaluate the function F and gradient G. On a return with */
/*             IFLAG=2, the user must provide the diagonal matrix Hk0. */
/**/
/*             The following negative values of IFLAG, detecting an error, */
/*             are possible: */
/**/
/*              IFLAG=-1  The line search routine MCSRCH failed. The */
/*                        parameter INFO provides more detailed information */
/*                        (see also the documentation of MCSRCH): */
/**/
/*                       INFO = 0  IMPROPER INPUT PARAMETERS. */
/**/
/*                       INFO = 2  RELATIVE WIDTH OF THE INTERVAL OF */
/*                                 UNCERTAINTY IS AT MOST XTOL. */
/**/
/*                       INFO = 3  MORE THAN 20 FUNCTION EVALUATIONS WERE */
/*                                 REQUIRED AT THE PRESENT ITERATION. */
/**/
/*                       INFO = 4  THE STEP IS TOO SMALL. */
/**/
/*                       INFO = 5  THE STEP IS TOO LARGE. */
/**/
/*                       INFO = 6  ROUNDING ERRORS PREVENT FURTHER PROGRESS. */
/*                                 THERE MAY NOT BE A STEP WHICH SATISFIES */
/*                                 THE SUFFICIENT DECREASE AND CURVATURE */
/*                                 CONDITIONS. TOLERANCES MAY BE TOO SMALL. */
/**/
/**/
/*              IFLAG=-2  The i-th diagonal element of the diagonal inverse */
/*                        Hessian approximation, given in DIAG, is not */
/*                        positive. */
/**/
/*              IFLAG=-3  Improper input parameters for LBFGS (N or M are */
/*                        not positive). */
/**/
/**/
/*    ON THE DRIVER: */
/*    The program that calls LBFGS must contain the declaration: */
/**/
/*                       EXTERNAL LB2 */
/**/
/*    LB2 is a BLOCK DATA that defines the default values of several */
/*    parameters described in the COMMON section. */
/**/
/**/
/*    COMMON: */
/**/
/*     The subroutine contains one common area, which the user may wish to */
/*    reference: */
/**/
/*<          COMMON /LB3/MP,LP,GTOL,STPMIN,STPMAX >*/
/**/
/*    MP  is an INT variable with default value 6. It is used as the */
/*        unit number for the printing of the monitoring information */
/*        controlled by IPRINT. */
/**/
/*    LP  is an INT variable with default value 6. It is used as the */
/*        unit number for the printing of error messages. This printing */
/*        may be suppressed by setting LP to a non-positive value. */
/**/
/*    GTOL is a DOUBLE PRECISION variable with default value 0.9, which */
/*        controls the accuracy of the line search routine MCSRCH. If the */
/*        function and gradient evaluations are inexpensive with respect */
/*        to the cost of the iteration (which is sometimes the case when */
/*        solving very large problems) it may be advantageous to set GTOL */
/*        to a small value. A typical small value is 0.1.  Restriction: */
/*        GTOL should be greater than 1.D-04. */
/**/
/*    STPMIN and STPMAX are non-negative DOUBLE PRECISION variables which */
/*        specify lower and uper bounds for the step in the line search. */
/*        Their default values are 1.D-20 and 1.D+20, respectively. These */
/*        values need not be modified unless the exponents are too large */
/*        for the machine being used, or unless the problem is extremely */
/*        badly scaled (in which case the exponents should be increased). */
/**/
/*  MACHINE DEPENDENCIES */
/**/
/*        The only variables that are machine-dependent are XTOL, */
/*        STPMIN and STPMAX. */
/**/
/*  GENERAL INFORMATION */
/**/
/*    Other routines called directly:  DAXPY, DDOT, LB1, MCSRCH */
/**/
/*    Input/Output  :  No input; diagnostic messages on unit MP and */
/*                     error messages on unit LP. */
/*     ******************************************************************************/

/*<       SUBROUTINE LBFGS(N,M,X,F,G,DIAGCO,DIAG,IPRINT,EPS,XTOL,W,IFLAG) >*/
int CLBFGS::lbfgs_(const int n, const int m, double *x, double
           *f, double *g, int *diagco, double *diag, int *iprint,
           const double eps, const double xtol, double *w, int *iflag)
{
    /* Initialized data */

    static double one = 1.;
    static double zero = 0.;


    int c__1 = 1;
    /* Format strings */
    /*
    static char fmt_245[] = "(/\002  GTOL IS LESS THAN OR EQUAL TO 1.D-04"
	    "\002,/\002 IT HAS BEEN RESET TO 9.D-01\002)";
    static char fmt_200[] = "(/\002 IFLAG= -1 \002,/\002 LINE SEARCH FAILED."
	    " SEE DOCUMENTATION OF ROUTINE MCSRCH\002,/\002 ERROR RETURN"
	    "\002\002 OF LINE SEARCH: INFO= \002,i2,/\002 POSSIBLE CAUSES: FU"
	    "NCTION OR GRADIENT ARE INCORRECT\002,/\002 OR INCORRECT TOLERANC"
	    "ES\002)";
    static char fmt_235[] = "(/\002 IFLAG= -2\002,/\002 THE\002,i5,\002-TH D"
	    "IAGONAL ELEMENT OF THE\002,/,\002 INVERSE HESSIAN APPROXIMATION "
	    "IS NOT POSITIVE\002)";
    static char fmt_240[] = "(/\002 IFLAG= -3\002,/\002 IMPROPER INPUT PARAM"
	    "ETERS (N OR M\002,\002 ARE NOT POSITIVE)\002)";

    */

    /* System generated locals */
    int i__1;
    double d__1;

    /* Local variables */
    static int i__, cp;
    static double sq, yr, ys, yy;
    static int npt;
    static double stp, stp1, beta;
    static int inmc;
    static int info, iscn, nfev, iycn, iter;
    static double ftol;
    static int nfun, ispt, iypt, bound;
//    static double gnorm;
    static int point;
//    static double xnorm_;
    static int finish;
    static int maxfev;

    /* Fortran I/O blocks */
/*    static cilist io___4 = { 0, 0, 0, fmt_245, 0 };*/
/*    static cilist io___30 = { 0, 0, 0, fmt_200, 0 };*/
/*    static cilist io___31 = { 0, 0, 0, fmt_235, 0 };*/
/*    static cilist io___32 = { 0, 0, 0, fmt_240, 0 };*/

    /* Parameter adjustments */
    --diag;
    --g;
    --x;
    --w;
    --iprint;

    /* Function Body */

/*<       IF(IFLAG.EQ.0) GO TO 10 >*/
    if (*iflag == 0) {
	goto L10;
    }
/*<       GO TO (172,100) IFLAG >*/
    switch (*iflag) {
	case 1:  goto L172;
	case 2:  goto L100;
    }
/*<   10  ITER= 0 >*/
L10:
    iter = 0;
/*<       IF(N.LE.0.OR.M.LE.0) GO TO 196 >*/
    if (n <= 0 || m <= 0) {
	goto L196;
    }
/*<       IF(GTOL.LE.1.D-04) THEN >*/
    if (gtol <= 1e-4) {
/*<         IF(LP.GT.0) WRITE(LP,245) >*/
      if (lp > 0) {
      /*   io___4.ciunit = lp;
	    s_wsfe(&io___4);
	    e_wsfe();*/
	}
/*<         GTOL=9.D-01 >*/
	gtol = .9;
/*<       ENDIF >*/
    }
/*<       NFUN= 1 >*/
    nfun = 1;
/*<       POINT= 0 >*/
    point = 0;
/*<       FINISH= .FALSE. >*/
    finish = LBFGS_FALSE;
/*<       IF(DIAGCO) THEN >*/
    if (*diagco) {
/*<          DO 30 I=1,N >*/
    i__1 = n;
	for (i__ = 1; i__ <= i__1; ++i__) {
/*<  30      IF (DIAG(I).LE.ZERO) GO TO 195 >*/
/* L30: */
	    if (diag[i__] <= zero) {
		goto L195;
	    }
	}
/*<       ELSE >*/
    } else {
/*<          DO 40 I=1,N >*/
    i__1 = n;
	for (i__ = 1; i__ <= i__1; ++i__) {
/*<  40      DIAG(I)= 1.0D0 >*/
/* L40: */
	    diag[i__] = 1.;
	}
/*<       ENDIF >*/
    }

/*     THE WORK VECTOR W IS DIVIDED AS FOLLOWS: */
/*     --------------------------------------- */
/*     THE FIRST N LOCATIONS ARE USED TO STORE THE GRADIENT AND */
/*         OTHER TEMPORARY INFORMATION. */
/*     LOCATIONS (N+1)...(N+M) STORE THE SCALARS RHO. */
/*     LOCATIONS (N+M+1)...(N+2M) STORE THE NUMBERS ALPHA USED */
/*         IN THE FORMULA THAT COMPUTES H*G. */
/*     LOCATIONS (N+2M+1)...(N+2M+NM) STORE THE LAST M SEARCH */
/*         STEPS. */
/*     LOCATIONS (N+2M+NM+1)...(N+2M+2NM) STORE THE LAST M */
/*         GRADIENT DIFFERENCES. */

/*     THE SEARCH STEPS AND GRADIENT DIFFERENCES ARE STORED IN A */
/*     CIRCULAR ORDER CONTROLLED BY THE PARAMETER POINT. */

/*<       ISPT= N+2*M >*/
    ispt = n + (m << 1);
/*<       IYPT= ISPT+M      >*/
    iypt = ispt + n * m;
/*<       DO 50 I=1,N >*/
    i__1 = n;
    for (i__ = 1; i__ <= i__1; ++i__) {
/*<  50   W(ISPT+I)= -G(I)*DIAG(I) >*/
/* L50: */
	w[ispt + i__] = -g[i__] * diag[i__];
    }
/*<       gnorm_= DSQRT(DDOT(N,G,1,G,1)) >*/
    gnorm_ = sqrt(ddot_(n, &g[1], &c__1, &g[1], &c__1));
/*<       STP1= ONE/gnorm_ >*/
    stp1 = one / gnorm_;

/*     PARAMETERS FOR LINE SEARCH ROUTINE */

/*<       FTOL= 1.0D-4 >*/
    ftol = 1e-4;
/*<       MAXFEV= 20 >*/
    maxfev = 20;

/*<        >*/

    //    if (iprint[1] >= 0) {
//	lb1_(&iprint[1], &iter, &nfun, &gnorm_, n, m, &x[1], f, &g[1], &stp, &
//		finish);
 //   }

/*    -------------------- */
/*     MAIN ITERATION LOOP */
/*    -------------------- */

/*<  80   ITER= ITER+1 >*/
L80:
    ++iter;
/*<       INFO=0 >*/
    info = 0;
/*<       BOUND=ITER-1 >*/
    bound = iter - 1;
/*<       IF(ITER.EQ.1) GO TO 165 >*/
    if (iter == 1) {
	goto L165;
    }
/*<       IF (ITER .GT. M)BOUND=M >*/
    if (iter > m) {
    bound = m;
    }

/*<          YS= DDOT(N,W(IYPT+NPT+1),1,W(ISPT+NPT+1),1) >*/
    ys = ddot_(n, &w[iypt + npt + 1], &c__1, &w[ispt + npt + 1], &c__1);
/*<       IF(.NOT.DIAGCO) THEN >*/
    if (! (*diagco)) {
/*<          YY= DDOT(N,W(IYPT+NPT+1),1,W(IYPT+NPT+1),1) >*/
	yy = ddot_(n, &w[iypt + npt + 1], &c__1, &w[iypt + npt + 1], &c__1);
/*<          DO 90 I=1,N >*/
    i__1 = n;
	for (i__ = 1; i__ <= i__1; ++i__) {
/*<    90    DIAG(I)= YS/YY >*/
/* L90: */
	    diag[i__] = ys / yy;
	}
/*<       ELSE >*/
    } else {
/*<          IFLAG=2 >*/
	*iflag = 2;
/*<          RETURN >*/
	return 0;
/*<       ENDIF >*/
    }
/*<  100  CONTINUE >*/
L100:
/*<       IF(DIAGCO) THEN >*/
    if (*diagco) {
/*<         DO 110 I=1,N >*/
    i__1 = n;
	for (i__ = 1; i__ <= i__1; ++i__) {
/*<  110    IF (DIAG(I).LE.ZERO) GO TO 195 >*/
/* L110: */
	    if (diag[i__] <= zero) {
		goto L195;
	    }
	}
/*<       ENDIF >*/
    }

/*     COMPUTE -H*G USING THE FORMULA GIVEN IN: Nocedal, J. 1980, */
/*     "Updating quasi-Newton matrices with limited storage", */
/*     Mathematics of Computation, Vol.24, No.151, pp. 773-782. */
/*     --------------------------------------------------------- */

/*<       CP= POINT >*/
    cp = point;
/*<       IF (POINT.EQ.0) CP=M >*/
    if (point == 0) {
    cp = m;
    }
/*<       W(N+CP)= ONE/YS >*/
    w[n + cp] = one / ys;
/*<       DO 112 I=1,N >*/
    i__1 = n;
    for (i__ = 1; i__ <= i__1; ++i__) {
/*<  112  W(I)= -G(I) >*/
/* L112: */
	w[i__] = -g[i__];
    }
/*<       CP= POINT >*/
    cp = point;
/*<       DO 125 I= 1,BOUND >*/
    i__1 = bound;
    for (i__ = 1; i__ <= i__1; ++i__) {
/*<          CP=CP-1 >*/
	--cp;
/*<          IF (CP.EQ. -1)CP=M-1 >*/
	if (cp == -1) {
        cp = m - 1;
	}
/*<          SQ= DDOT(N,W(ISPT+CP*N+1),1,W,1) >*/
    sq = ddot_(n, &w[ispt + cp * n + 1], &c__1, &w[1], &c__1);
/*<          INMC=N+M+CP+1 >*/
    inmc = n + m + cp + 1;
/*<          IYCN=IYPT+CP >*/
    iycn = iypt + cp * n;
/*<          W(INMC)= W(N+CP+1)*SQ >*/
    w[inmc] = w[n + cp + 1] * sq;
/*<          CALL DAXPY(N,-W(INMC),W(IYCN+1),1,W,1) >*/
	d__1 = -w[inmc];
	daxpy_(n, &d__1, &w[iycn + 1], &c__1, &w[1], &c__1);
/*<  125  CONTINUE >*/
/* L125: */
    }

/*<       DO 130 I=1,N >*/
    i__1 = n;
    for (i__ = 1; i__ <= i__1; ++i__) {
/*<  130  W(I)=DIAG(I)*W(I) >*/
/* L130: */
	w[i__] = diag[i__] * w[i__];
    }

/*<       DO 145 I=1,BOUND >*/
    i__1 = bound;
    for (i__ = 1; i__ <= i__1; ++i__) {
/*<          YR= DDOT(N,W(IYPT+CP*N+1),1,W,1) >*/
    yr = ddot_(n, &w[iypt + cp * n + 1], &c__1, &w[1], &c__1);
/*<          BETA= W(N+CP+1)*YR >*/
    beta = w[n + cp + 1] * yr;
/*<          INMC=N+M+CP+1 >*/
    inmc = n + m + cp + 1;
/*<          BETA= W(INMC)-BETA >*/
	beta = w[inmc] - beta;
/*<          ISCN=ISPT+CP*N >*/
    iscn = ispt + cp * n;
/*<          CALL DAXPY(N,BETA,W(ISCN+1),1,W,1) >*/
	daxpy_(n, &beta, &w[iscn + 1], &c__1, &w[1], &c__1);
/*<          CP=CP+1 >*/
	++cp;
/*<          IF (CP.EQ.M)CP=0 >*/
    if (cp == m) {
	    cp = 0;
	}
/*<  145  CONTINUE >*/
/* L145: */
    }

/*     STORE THE NEW SEARCH DIRECTION */
/*     ------------------------------ */

/*<        DO 160 I=1,N >*/
    i__1 = n;
    for (i__ = 1; i__ <= i__1; ++i__) {
/*<  160   W(ISPT+POINT*N+I)= W(I) >*/
/* L160: */
    w[ispt + point * n + i__] = w[i__];
    }

/*     OBTAIN THE ONE-DIMENSIONAL MINIMIZER OF THE FUNCTION */
/*     BY USING THE LINE SEARCH ROUTINE MCSRCH */
/*     ---------------------------------------------------- */
/*<  165  NFEV=0 >*/
L165:
    nfev = 0;
/*<       STP=ONE >*/
    stp = one;
/*<       IF (ITER.EQ.1) STP=STP1 >*/
    if (iter == 1) {
	stp = stp1;
    }
/*<       DO 170 I=1,N >*/
    i__1 = n;
    for (i__ = 1; i__ <= i__1; ++i__) {
/*<  170  W(I)=G(I) >*/
/* L170: */
	w[i__] = g[i__];
    }
/*<  172  CONTINUE >*/
L172:
/*<        >*/
    mcsrch_(n, &x[1], f, &g[1], &w[ispt + point * n + 1], &stp, &ftol, xtol,
	    &maxfev, &info, &nfev, &diag[1]);
/*<       IF (INFO .EQ. -1) THEN >*/
    if (info == -1) {
/*<         IFLAG=1 >*/
	*iflag = 1;
/*<         RETURN >*/
	return 0;
/*<       ENDIF >*/
    }
/*<       IF (INFO .NE. 1) GO TO 190 >*/
    if (info != 1) {
	goto L190;
    }
/*<       NFUN= NFUN + NFEV >*/
    nfun += nfev;

/*     COMPUTE THE NEW STEP AND GRADIENT CHANGE */
/*     ----------------------------------------- */

/*<       NPT=POINT*N >*/
    npt = point * n;
/*<       DO 175 I=1,N >*/
    i__1 = n;
    for (i__ = 1; i__ <= i__1; ++i__) {
/*<       W(ISPT+NPT+I)= STP*W(ISPT+NPT+I) >*/
	w[ispt + npt + i__] = stp * w[ispt + npt + i__];
/*<  175  W(IYPT+NPT+I)= G(I)-W(I) >*/
/* L175: */
	w[iypt + npt + i__] = g[i__] - w[i__];
    }
/*<       POINT=POINT+1 >*/
    ++point;
/*<       IF (POINT.EQ.M)POINT=0 >*/
    if (point == m) {
	point = 0;
    }

/*     TERMINATION TEST */
/*     ---------------- */

/*<       gnorm_= DSQRT(DDOT(N,G,1,G,1)) >*/
    gnorm_ = sqrt(ddot_(n, &g[1], &c__1, &g[1], &c__1));
/*<       xnorm_= DSQRT(DDOT(N,X,1,X,1)) >*/
    xnorm_ = sqrt(ddot_(n, &x[1], &c__1, &x[1], &c__1));
/*<       xnorm_= DMAX1(1.0D0,xnorm_) >*/
    xnorm_ = LBFGS_MAX(1.,xnorm_);
/*<       IF (gnorm_/xnorm_ .LE. EPS) FINISH=.TRUE. >*/
    if (gnorm_ / xnorm_ <= eps) {
    finish = LBFGS_TRUE;
    }

/*<        >*/
//    if (iprint[1] >= 0) {
//	lb1_(&iprint[1], &iter, &nfun, &gnorm, n, m, &x[1], f, &g[1], &stp, &
//		finish);
//    }
/*<       IF (FINISH) THEN >*/
    if (finish) {
/*<          IFLAG=0 >*/
	*iflag = 0;
/*<          RETURN >*/
	return 0;
/*<       ENDIF >*/
    }
/*<       GO TO 80 >*/
    goto L80;

/*     ------------------------------------------------------------ */
/*     END OF MAIN ITERATION LOOP. ERROR EXITS. */
/*     ------------------------------------------------------------ */

/*<  190  IFLAG=-1 >*/
L190:
    *iflag = -1;
/*<       IF(LP.GT.0) WRITE(LP,200) INFO >*/
    if (lp > 0) {
/*	io___30.ciunit = lp;*/
/*	s_wsfe(&io___30);*/
/*	do_fio(&c__1, (char *)&info, (ftnlen)sizeof(int));*/
/*	e_wsfe();*/
    }
/*<       RETURN >*/
    return 0;
/*<  195  IFLAG=-2 >*/
L195:
    *iflag = -2;
/*<       IF(LP.GT.0) WRITE(LP,235) I >*/
    if (lp > 0) {
/*	io___31.ciunit = lp;*/
/*	s_wsfe(&io___31);*/
/*	do_fio(&c__1, (char *)&i__, (ftnlen)sizeof(int));*/
/*	e_wsfe();*/
    }
/*<       RETURN >*/
    return 0;
/*<  196  IFLAG= -3 >*/
L196:
    *iflag = -3;
/*<       IF(LP.GT.0) WRITE(LP,240) >*/
    if (lp > 0) {
/*	io___32.ciunit = lb3_1.lp;*/
/*	s_wsfe(&io___32);*/
/*	e_wsfe();*/
    }

/*<       RETURN >*/
    return 0;
/*<       END >*/
} /* lbfgs_ */




/*=====================================================================
  =======================================================================*/
/*<       subroutine daxpy(n,da,dx,incx,dy,incy) >*/
int CLBFGS::daxpy_(int n, double *da, double *dx, int *incx, double *dy, int *incy)
{
    /* System generated locals */
    int i__1;

    /* Local variables */
    static int i__, m, ix, iy, mp1;


/*     constant times a vector plus a vector. */
/*     uses unrolled loops for increments equal to one. */
/*     jack dongarra, linpack, 3/11/78. */

/*<       double precision dx(1),dy(1),da >*/
/*<       int i,incx,incy,ix,iy,m,mp1,n >*/

/*<       if(n.le.0)return >*/
    /* Parameter adjustments */
    --dy;
    --dx;

    /* Function Body */
    if (n <= 0) {
	return 0;
    }
/*<       if (da .eq. 0.0d0) return >*/
    if (*da == 0.) {
	return 0;
    }
/*<       if(incx.eq.1.and.incy.eq.1)go to 20 >*/
    if (*incx == 1 && *incy == 1) {
	goto L20;
    }

/*        code for unequal increments or equal increments */
/*          not equal to 1 */

/*<       ix = 1 >*/
    ix = 1;
/*<       iy = 1 >*/
    iy = 1;
/*<       if(incx.lt.0)ix = (-n+1)*incx + 1 >*/
    if (*incx < 0) {
    ix = (-(n) + 1) * *incx + 1;
    }
/*<       if(incy.lt.0)iy = (-n+1)*incy + 1 >*/
    if (*incy < 0) {
    iy = (-(n) + 1) * *incy + 1;
    }
/*<       do 10 i = 1,n >*/
    i__1 = n;
    for (i__ = 1; i__ <= i__1; ++i__) {
/*<         dy(iy) = dy(iy) + da*dx(ix) >*/
	dy[iy] += *da * dx[ix];
/*<         ix = ix + incx >*/
	ix += *incx;
/*<         iy = iy + incy >*/
	iy += *incy;
/*<    10 continue >*/
/* L10: */
    }
/*<       return >*/
    return 0;

/*        code for both increments equal to 1 */


/*        clean-up loop */

/*<    20 m = mod(n,4) >*/
L20:
    m = n % 4;
/*<       if( m .eq. 0 ) go to 40 >*/
    if (m == 0) {
	goto L40;
    }
/*<       do 30 i = 1,m >*/
    i__1 = m;
    for (i__ = 1; i__ <= i__1; ++i__) {
/*<         dy(i) = dy(i) + da*dx(i) >*/
	dy[i__] += *da * dx[i__];
/*<    30 continue >*/
/* L30: */
    }
/*<       if( n .lt. 4 ) return >*/
    if (n < 4) {
	return 0;
    }
/*<    40 mp1 = m + 1 >*/
L40:
    mp1 = m + 1;
/*<       do 50 i = mp1,n,4 >*/
    i__1 = n;
    for (i__ = mp1; i__ <= i__1; i__ += 4) {
/*<         dy(i) = dy(i) + da*dx(i) >*/
	dy[i__] += *da * dx[i__];
/*<         dy(i + 1) = dy(i + 1) + da*dx(i + 1) >*/
	dy[i__ + 1] += *da * dx[i__ + 1];
/*<         dy(i + 2) = dy(i + 2) + da*dx(i + 2) >*/
	dy[i__ + 2] += *da * dx[i__ + 2];
/*<         dy(i + 3) = dy(i + 3) + da*dx(i + 3) >*/
	dy[i__ + 3] += *da * dx[i__ + 3];
/*<    50 continue >*/
/* L50: */
    }
/*<       return >*/
    return 0;
/*<       end >*/
} /* daxpy_ */



/*=====================================================================
  =======================================================================*/
/*<       double precision function ddot(n,dx,incx,dy,incy) >*/
double CLBFGS::ddot_(int n, double *dx, int *incx, double *dy, int *incy)
{
    /* System generated locals */
    int i__1;
    double ret_val;

    /* Local variables */
    static int i__, m, ix, iy, mp1;
    static double dtemp;

/*     forms the dot product of two vectors. */
/*     uses unrolled loops for increments equal to one. */
/*     jack dongarra, linpack, 3/11/78. */

/*<       double precision dx(1),dy(1),dtemp >*/
/*<       int i,incx,incy,ix,iy,m,mp1,n >*/

/*<       ddot = 0.0d0 >*/
    /* Parameter adjustments */
    --dy;
    --dx;

    /* Function Body */
    ret_val = 0.;
/*<       dtemp = 0.0d0 >*/
    dtemp = 0.;
/*<       if(n.le.0)return >*/
    if (n <= 0) {
	return ret_val;
    }
/*<       if(incx.eq.1.and.incy.eq.1)go to 20 >*/
    if (*incx == 1 && *incy == 1) {
	goto L20;
    }

/*        code for unequal increments or equal increments */
/*          not equal to 1 */

/*<       ix = 1 >*/
    ix = 1;
/*<       iy = 1 >*/
    iy = 1;
/*<       if(incx.lt.0)ix = (-n+1)*incx + 1 >*/
    if (*incx < 0) {
    ix = (-(n) + 1) * *incx + 1;
    }
/*<       if(incy.lt.0)iy = (-n+1)*incy + 1 >*/
    if (*incy < 0) {
    iy = (-(n) + 1) * *incy + 1;
    }
/*<       do 10 i = 1,n >*/
    i__1 = n;
    for (i__ = 1; i__ <= i__1; ++i__) {
/*<         dtemp = dtemp + dx(ix)*dy(iy) >*/
	dtemp += dx[ix] * dy[iy];
/*<         ix = ix + incx >*/
	ix += *incx;
/*<         iy = iy + incy >*/
	iy += *incy;
/*<    10 continue >*/
/* L10: */
    }
/*<       ddot = dtemp >*/
    ret_val = dtemp;
/*<       return >*/
    return ret_val;

/*        code for both increments equal to 1 */


/*        clean-up loop */

/*<    20 m = mod(n,5) >*/
L20:
    m = n % 5;
/*<       if( m .eq. 0 ) go to 40 >*/
    if (m == 0) {
	goto L40;
    }
/*<       do 30 i = 1,m >*/
    i__1 = m;
    for (i__ = 1; i__ <= i__1; ++i__) {
/*<         dtemp = dtemp + dx(i)*dy(i) >*/
	dtemp += dx[i__] * dy[i__];
/*<    30 continue >*/
/* L30: */
    }
/*<       if( n .lt. 5 ) go to 60 >*/
    if (n < 5) {
	goto L60;
    }
/*<    40 mp1 = m + 1 >*/
L40:
    mp1 = m + 1;
/*<       do 50 i = mp1,n,5 >*/
    i__1 = n;
    for (i__ = mp1; i__ <= i__1; i__ += 5) {
/*<        >*/
	dtemp = dtemp + dx[i__] * dy[i__] + dx[i__ + 1] * dy[i__ + 1] + dx[
		i__ + 2] * dy[i__ + 2] + dx[i__ + 3] * dy[i__ + 3] + dx[i__ + 
		4] * dy[i__ + 4];
/*<    50 continue >*/
/* L50: */
    }
/*<    60 ddot = dtemp >*/
L60:
    ret_val = dtemp;
/*<       return >*/
    return ret_val;
/*<       end >*/
} /* ddot_ */


/*=====================================================================
     LINE SEARCH ROUTINE MCSRCH 
  =======================================================================*/

/*<       SUBROUTINE MCSRCH(N,X,F,G,S,STP,FTOL,XTOL,MAXFEV,INFO,NFEV,WA) >*/
int CLBFGS::mcsrch_(int n, double *x, double *f,
            double *g, double *s, double *stp, double *ftol, 
            double xtol, int *maxfev, int *info, int *nfev,
            double *wa)
{
    /* Initialized data */

    static double p5 = .5;
    static double p66 = .66;
    static double xtrapf = 4.;
    static double zero = 0.;

    /* Format strings */
/*    static char fmt_15[] = "(/\002  THE SEARCH DIRECTION IS NOT A DESCENT DI"*/
/*	    "RECTION\002)";*/

    /* System generated locals */
    int i__1;
    double d__1;

    /* Builtin functions */
/*    int s_wsfe(cilist *), e_wsfe(void);*/

    /* Local variables */
    static int j;
    static double dg, fm, fx, fy, dgm, dgx, dgy, fxm, fym, stx, sty, dgxm,
         dgym;
    static int infoc;
    static double finit, width, stmin, stmax;
    static int stage1;
    static double width1, ftest1;
    static int brackt;
    static double dginit, dgtest;

/*                     SUBROUTINE MCSRCH */
/*     A slight modification of the subroutine CSRCH of More' and Thuente. */
/*     The changes are to allow reverse communication, and do not affect */
/*     the performance of the routine. */
/* */
/*     THE PURPOSE OF MCSRCH IS TO FIND A STEP WHICH SATISFIES */
/*     A SUFFICIENT DECREASE CONDITION AND A CURVATURE CONDITION. */
    /* */
/*     AT EACH STAGE THE SUBROUTINE UPDATES AN INTERVAL OF */
/*     UNCERTAINTY WITH ENDPOINTS STX AND STY. THE INTERVAL OF */
/*     UNCERTAINTY IS INITIALLY CHOSEN SO THAT IT CONTAINS A */
/*     MINIMIZER OF THE MODIFIED FUNCTION */
/* */
/*          F(X+STP*S) - F(X) - FTOL*STP*(GRADF(X)'S). */
/* */
/*     IF A STEP IS OBTAINED FOR WHICH THE MODIFIED FUNCTION */
/*     HAS A NONPOSITIVE FUNCTION VALUE AND NONNEGATIVE DERIVATIVE, */
/*     THEN THE INTERVAL OF UNCERTAINTY IS CHOSEN SO THAT IT */
/*     CONTAINS A MINIMIZER OF F(X+STP*S). */

/*     THE ALGORITHM IS DESIGNED TO FIND A STEP WHICH SATISFIES */
/*     THE SUFFICIENT DECREASE CONDITION */

/*           F(X+STP*S) .LE. F(X) + FTOL*STP*(GRADF(X)'S), */

/*     AND THE CURVATURE CONDITION */

/*           LBFGS_ABS(GRADF(X+STP*S)'S)) .LE. GTOL*LBFGS_ABS(GRADF(X)'S). */

/*     IF FTOL IS LESS THAN GTOL AND IF, FOR EXAMPLE, THE FUNCTION */
/*     IS BOUNDED BELOW, THEN THERE IS ALWAYS A STEP WHICH SATISFIES */
/*     BOTH CONDITIONS. IF NO STEP CAN BE FOUND WHICH SATISFIES BOTH */
/*     CONDITIONS, THEN THE ALGORITHM USUALLY STOPS WHEN ROUNDING */
/*     ERRORS PREVENT FURTHER PROGRESS. IN THIS CASE STP ONLY */
/*     SATISFIES THE SUFFICIENT DECREASE CONDITION. */

/*     THE SUBROUTINE STATEMENT IS */

/*        SUBROUTINE MCSRCH(N,X,F,G,S,STP,FTOL,XTOL, MAXFEV,INFO,NFEV,WA) */
/*     WHERE */

/*       N IS A POSITIVE INT INPUT VARIABLE SET TO THE NUMBER */
/*         OF VARIABLES. */

/*       X IS AN ARRAY OF LENGTH N. ON INPUT IT MUST CONTAIN THE */
/*         BASE POINT FOR THE LINE SEARCH. ON OUTPUT IT CONTAINS */
/*         X + STP*S. */

/*       F IS A VARIABLE. ON INPUT IT MUST CONTAIN THE VALUE OF F */
/*         AT X. ON OUTPUT IT CONTAINS THE VALUE OF F AT X + STP*S. */

/*       G IS AN ARRAY OF LENGTH N. ON INPUT IT MUST CONTAIN THE */
/*         GRADIENT OF F AT X. ON OUTPUT IT CONTAINS THE GRADIENT */
/*         OF F AT X + STP*S. */

/*       S IS AN INPUT ARRAY OF LENGTH N WHICH SPECIFIES THE */
/*         SEARCH DIRECTION. */

/*       STP IS A NONNEGATIVE VARIABLE. ON INPUT STP CONTAINS AN */
/*         INITIAL ESTIMATE OF A SATISFACTORY STEP. ON OUTPUT */
/*         STP CONTAINS THE FINAL ESTIMATE. */

/*       FTOL AND GTOL ARE NONNEGATIVE INPUT VARIABLES. (In this reverse */
/*         communication implementation GTOL is defined in a COMMON */
/*         statement.) TERMINATION OCCURS WHEN THE SUFFICIENT DECREASE */
/*         CONDITION AND THE DIRECTIONAL DERIVATIVE CONDITION ARE */
/*         SATISFIED. */

/*       XTOL IS A NONNEGATIVE INPUT VARIABLE. TERMINATION OCCURS */
/*         WHEN THE RELATIVE WIDTH OF THE INTERVAL OF UNCERTAINTY */
/*         IS AT MOST XTOL. */

/*       STPMIN AND STPMAX ARE NONNEGATIVE INPUT VARIABLES WHICH */
/*         SPECIFY LOWER AND UPPER BOUNDS FOR THE STEP. (In this reverse */
/*         communication implementatin they are defined in a COMMON */
/*         statement). */

/*       MAXFEV IS A POSITIVE INT INPUT VARIABLE. TERMINATION */
/*         OCCURS WHEN THE NUMBER OF CALLS TO FCN IS AT LEAST */
/*         MAXFEV BY THE END OF AN ITERATION. */

/*       INFO IS AN INT OUTPUT VARIABLE SET AS FOLLOWS: */

/*         INFO = 0  IMPROPER INPUT PARAMETERS. */

/*         INFO =-1  A RETURN IS MADE TO COMPUTE THE FUNCTION AND GRADIENT. */

/*         INFO = 1  THE SUFFICIENT DECREASE CONDITION AND THE */
/*                   DIRECTIONAL DERIVATIVE CONDITION HOLD. */

/*         INFO = 2  RELATIVE WIDTH OF THE INTERVAL OF UNCERTAINTY */
/*                   IS AT MOST XTOL. */

/*         INFO = 3  NUMBER OF CALLS TO FCN HAS REACHED MAXFEV. */

/*         INFO = 4  THE STEP IS AT THE LOWER BOUND STPMIN. */

/*         INFO = 5  THE STEP IS AT THE UPPER BOUND STPMAX. */

/*         INFO = 6  ROUNDING ERRORS PREVENT FURTHER PROGRESS. */
/*                   THERE MAY NOT BE A STEP WHICH SATISFIES THE */
/*                   SUFFICIENT DECREASE AND CURVATURE CONDITIONS. */
/*                   TOLERANCES MAY BE TOO SMALL. */

/*       NFEV IS AN INT OUTPUT VARIABLE SET TO THE NUMBER OF */
/*         CALLS TO FCN. */

/*       WA IS A WORK ARRAY OF LENGTH N. */

/*     SUBPROGRAMS CALLED */

/*       MCSTEP */

/*       FORTRAN-SUPPLIED...ABS,MAX,MIN */

/*     ARGONNE NATIONAL LABORATORY. MINPACK PROJECT. JUNE 1983 */
/*     JORGE J. MORE', DAVID J. THUENTE */

/*     ********** */
/*<       INT INFOC,J >*/
/*<       INT BRACKT,STAGE1 >*/
/*<        >*/
/*<       DATA P5,P66,XTRAPF,ZERO /0.5D0,0.66D0,4.0D0,0.0D0/ >*/
    /* Parameter adjustments */
    --wa;
    --s;
    --g;
    --x;

    /* Function Body */
/*<       IF(INFO.EQ.-1) GO TO 45 >*/
    if (*info == -1) {
	goto L45;
    }
/*<       INFOC = 1 >*/
    infoc = 1;

/*     CHECK THE INPUT PARAMETERS FOR ERRORS. */

/*<        >*/
    if (n <= 0 || *stp <= zero || *ftol < zero || gtol < zero || xtol
	    < zero || stpmin < zero || stpmax < stpmin || *
	    maxfev <= 0) {
	return 0;
    }

/*     COMPUTE THE INITIAL GRADIENT IN THE SEARCH DIRECTION */
/*     AND CHECK THAT S IS A DESCENT DIRECTION. */

/*<       DGINIT = ZERO >*/
    dginit = zero;
/*<       DO 10 J = 1, N >*/
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
/*<          DGINIT = DGINIT + G(J)*S(J) >*/
	dginit += g[j] * s[j];
/*<    10    CONTINUE >*/
/* L10: */
    }
/*<       IF (DGINIT .GE. ZERO) then >*/
    if (dginit >= zero) {
/*<          write(LP,15) >*/
/*	io___71.ciunit = lp;*/
/*	s_wsfe(&io___71);*/
/*	e_wsfe();*/
/*<    15    FORMAT(/'  THE SEARCH DIRECTION IS NOT A DESCENT DIRECTION') >*/
/*<          RETURN >*/
	return 0;
/*<          ENDIF >*/
    }

/*     INITIALIZE LOCAL VARIABLES. */

/*<       BRACKT = .FALSE. >*/
    brackt = LBFGS_FALSE;
/*<       STAGE1 = .TRUE. >*/
    stage1 = LBFGS_TRUE;
/*<       NFEV = 0 >*/
    *nfev = 0;
/*<       FINIT = F >*/
    finit = *f;
/*<       DGTEST = FTOL*DGINIT >*/
    dgtest = *ftol * dginit;
/*<       WIDTH = STPMAX - STPMIN >*/
    width = stpmax - stpmin;
/*<       WIDTH1 = WIDTH/P5 >*/
    width1 = width / p5;
/*<       DO 20 J = 1, N >*/
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
/*<          WA(J) = X(J) >*/
	wa[j] = x[j];
/*<    20    CONTINUE >*/
/* L20: */
    }

/*     THE VARIABLES STX, FX, DGX CONTAIN THE VALUES OF THE STEP, */
/*     FUNCTION, AND DIRECTIONAL DERIVATIVE AT THE BEST STEP. */
/*     THE VARIABLES STY, FY, DGY CONTAIN THE VALUE OF THE STEP, */
/*     FUNCTION, AND DERIVATIVE AT THE OTHER ENDPOINT OF */
/*     THE INTERVAL OF UNCERTAINTY. */
/*     THE VARIABLES STP, F, DG CONTAIN THE VALUES OF THE STEP, */
/*     FUNCTION, AND DERIVATIVE AT THE CURRENT STEP. */

/*<       STX = ZERO >*/
    stx = zero;
/*<       FX = FINIT >*/
    fx = finit;
/*<       DGX = DGINIT >*/
    dgx = dginit;
/*<       STY = ZERO >*/
    sty = zero;
/*<       FY = FINIT >*/
    fy = finit;
/*<       DGY = DGINIT >*/
    dgy = dginit;

/*     START OF ITERATION. */

/*<    30 CONTINUE >*/
L30:

/*        SET THE MINIMUM AND MAXIMUM STEPS TO CORRESPOND */
/*        TO THE PRESENT INTERVAL OF UNCERTAINTY. */

/*<          IF (BRACKT) THEN >*/
    if (brackt) {
/*<             STMIN = LBFGS_MIN(STX,STY) >*/
	stmin = LBFGS_MIN(stx,sty);
/*<             STMAX = LBFGS_MAX(STX,STY) >*/
	stmax = LBFGS_MAX(stx,sty);
/*<          ELSE >*/
    } else {
/*<             STMIN = STX >*/
	stmin = stx;
/*<             STMAX = STP + XTRAPF*(STP - STX) >*/
	stmax = *stp + xtrapf * (*stp - stx);
/*<             END IF >*/
    }

/*        FORCE THE STEP TO BE WITHIN THE BOUNDS STPMAX AND STPMIN. */

/*<          STP = LBFGS_MAX(STP,STPMIN) >*/
    *stp = LBFGS_MAX(*stp,stpmin);
/*<          STP = LBFGS_MIN(STP,STPMAX) >*/
    *stp = LBFGS_MIN(*stp,stpmax);

/*        IF AN UNUSUAL TERMINATION IS TO OCCUR THEN LET */
/*        STP BE THE LOWEST POINT OBTAINED SO FAR. */

/*<        >*/
    if (brackt && (*stp <= stmin || *stp >= stmax) || *nfev >= *maxfev - 1 || 
        infoc == 0 || brackt && stmax - stmin <= xtol * stmax) {
	*stp = stx;
    }

/*        EVALUATE THE FUNCTION AND GRADIENT AT STP */
/*        AND COMPUTE THE DIRECTIONAL DERIVATIVE. */
/*        We return to main program to obtain F and G. */

/*<          DO 40 J = 1, N >*/
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
/*<             X(J) = WA(J) + STP*S(J) >*/
	x[j] = wa[j] + *stp * s[j];
/*<    40       CONTINUE >*/
/* L40: */
    }
/*<          INFO=-1 >*/
    *info = -1;
/*<          RETURN >*/
    return 0;

/*<    45    INFO=0 >*/
L45:
    *info = 0;
/*<          NFEV = NFEV + 1 >*/
    ++(*nfev);
/*<          DG = ZERO >*/
    dg = zero;
/*<          DO 50 J = 1, N >*/
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
/*<             DG = DG + G(J)*S(J) >*/
	dg += g[j] * s[j];
/*<    50       CONTINUE >*/
/* L50: */
    }
/*<          FTEST1 = FINIT + STP*DGTEST >*/
    ftest1 = finit + *stp * dgtest;

/*        TEST FOR CONVERGENCE. */

/*<        >*/
    if (brackt && (*stp <= stmin || *stp >= stmax) || infoc == 0) {
	*info = 6;
    }
/*<        >*/
    if (*stp == stpmax && *f <= ftest1 && dg <= dgtest) {
	*info = 5;
    }
/*<        >*/
    if (*stp == stpmin && (*f > ftest1 || dg >= dgtest)) {
	*info = 4;
    }
/*<          IF (NFEV .GE. MAXFEV) INFO = 3 >*/
    if (*nfev >= *maxfev) {
	*info = 3;
    }
/*<          IF (BRACKT .AND. STMAX-STMIN .LE. XTOL*STMAX) INFO = 2 >*/
    if (brackt && stmax - stmin <= xtol * stmax) {
	*info = 2;
    }
/*<          IF (F .LE. FTEST1 .AND. LBFGS_ABS(DG) .LE. GTOL*(-DGINIT)) INFO = 1 >*/
    if (*f <= ftest1 && LBFGS_ABS(dg) <= gtol * (-dginit)) {
	*info = 1;
    }

/*        CHECK FOR TERMINATION. */

/*<          IF (INFO .NE. 0) RETURN >*/
    if (*info != 0) {
	return 0;
    }

/*        IN THE FIRST STAGE WE SEEK A STEP FOR WHICH THE MODIFIED */
/*        FUNCTION HAS A NONPOSITIVE VALUE AND NONNEGATIVE DERIVATIVE. */

/*<        >*/
    if (stage1 && *f <= ftest1 && dg >= LBFGS_MIN(*ftol,gtol) * dginit) {
    stage1 = LBFGS_FALSE;
    }

/*        A MODIFIED FUNCTION IS USED TO PREDICT THE STEP ONLY IF */
/*        WE HAVE NOT OBTAINED A STEP FOR WHICH THE MODIFIED */
/*        FUNCTION HAS A NONPOSITIVE FUNCTION VALUE AND NONNEGATIVE */
/*        DERIVATIVE, AND IF A LOWER FUNCTION VALUE HAS BEEN */
/*        OBTAINED BUT THE DECREASE IS NOT SUFFICIENT. */

/*<          IF (STAGE1 .AND. F .LE. FX .AND. F .GT. FTEST1) THEN >*/
    if (stage1 && *f <= fx && *f > ftest1) {

/*           DEFINE THE MODIFIED FUNCTION AND DERIVATIVE VALUES. */

/*<             FM = F - STP*DGTEST >*/
	fm = *f - *stp * dgtest;
/*<             FXM = FX - STX*DGTEST >*/
	fxm = fx - stx * dgtest;
/*<             FYM = FY - STY*DGTEST >*/
	fym = fy - sty * dgtest;
/*<             DGM = DG - DGTEST >*/
	dgm = dg - dgtest;
/*<             DGXM = DGX - DGTEST >*/
	dgxm = dgx - dgtest;
/*<             DGYM = DGY - DGTEST >*/
	dgym = dgy - dgtest;

/*           CALL CSTEP TO UPDATE THE INTERVAL OF UNCERTAINTY */
/*           AND TO COMPUTE THE NEW STEP. */

/*<        >*/
	mcstep_(&stx, &fxm, &dgxm, &sty, &fym, &dgym, stp, &fm, &dgm, &brackt,
		 &stmin, &stmax, &infoc);

/*           RESET THE FUNCTION AND GRADIENT VALUES FOR F. */

/*<             FX = FXM + STX*DGTEST >*/
	fx = fxm + stx * dgtest;
/*<             FY = FYM + STY*DGTEST >*/
	fy = fym + sty * dgtest;
/*<             DGX = DGXM + DGTEST >*/
	dgx = dgxm + dgtest;
/*<             DGY = DGYM + DGTEST >*/
	dgy = dgym + dgtest;
/*<          ELSE >*/
    } else {

/*           CALL MCSTEP TO UPDATE THE INTERVAL OF UNCERTAINTY */
/*           AND TO COMPUTE THE NEW STEP. */

/*<        >*/
	mcstep_(&stx, &fx, &dgx, &sty, &fy, &dgy, stp, f, &dg, &brackt, &
		stmin, &stmax, &infoc);
/*<             END IF >*/
    }

/*        FORCE A SUFFICIENT DECREASE IN THE SIZE OF THE */
/*        INTERVAL OF UNCERTAINTY. */

/*<          IF (BRACKT) THEN >*/
    if (brackt) {
/*<        >*/
	if ((d__1 = sty - stx, LBFGS_ABS(d__1)) >= p66 * width1) {
	    *stp = stx + p5 * (sty - stx);
	}
/*<             WIDTH1 = WIDTH >*/
	width1 = width;
/*<             WIDTH = LBFGS_ABS(STY-STX) >*/
	width = (d__1 = sty - stx, LBFGS_ABS(d__1));
/*<             END IF >*/
    }

/*        END OF ITERATION. */

/*<          GO TO 30 >*/
    goto L30;

/*     LAST LINE OF SUBROUTINE MCSRCH. */

/*<       END >*/
} /* mcsrch_ */




/*=========================================================================
  =========================================================================*/

int CLBFGS::mcstep_(double *stx, double *fx, double *dx,
             double *sty, double *fy, double *dy, double *stp, 
             double *fp, double *dp, int *brackt, double *stpmin, 
             double *stpmax, int *info)
{
    /* System generated locals */
    double d__1, d__2, d__3;

    /* Builtin functions */
    double sqrt(double);

    /* Local variables */
    static double p, q, r__, s, sgnd, stpc, stpf, stpq, gamma, theta;
    static int bound;

/*<       INT INFO >*/
/*<       DOUBLE PRECISION STX,FX,DX,STY,FY,DY,STP,FP,DP,STPMIN,STPMAX >*/
/*<       INT BRACKT,BOUND >*/

/*     SUBROUTINE MCSTEP */

/*     THE PURPOSE OF MCSTEP IS TO COMPUTE A SAFEGUARDED STEP FOR */
/*     A LINESEARCH AND TO UPDATE AN INTERVAL OF UNCERTAINTY FOR */
/*     A MINIMIZER OF THE FUNCTION. */

/*     THE PARAMETER STX CONTAINS THE STEP WITH THE LEAST FUNCTION */
/*     VALUE. THE PARAMETER STP CONTAINS THE CURRENT STEP. IT IS */
/*     ASSUMED THAT THE DERIVATIVE AT STX IS NEGATIVE IN THE */
/*     DIRECTION OF THE STEP. IF BRACKT IS SET TRUE THEN A */
/*     MINIMIZER HAS BEEN BRACKETED IN AN INTERVAL OF UNCERTAINTY */
/*     WITH ENDPOINTS STX AND STY. */

/*     THE SUBROUTINE STATEMENT IS */

/*       SUBROUTINE MCSTEP(STX,FX,DX,STY,FY,DY,STP,FP,DP,BRACKT, */
/*                        STPMIN,STPMAX,INFO) */

/*     WHERE */

/*       STX, FX, AND DX ARE VARIABLES WHICH SPECIFY THE STEP, */
/*         THE FUNCTION, AND THE DERIVATIVE AT THE BEST STEP OBTAINED */
/*         SO FAR. THE DERIVATIVE MUST BE NEGATIVE IN THE DIRECTION */
/*         OF THE STEP, THAT IS, DX AND STP-STX MUST HAVE OPPOSITE */
/*         SIGNS. ON OUTPUT THESE PARAMETERS ARE UPDATED APPROPRIATELY. */

/*       STY, FY, AND DY ARE VARIABLES WHICH SPECIFY THE STEP, */
/*         THE FUNCTION, AND THE DERIVATIVE AT THE OTHER ENDPOINT OF */
/*         THE INTERVAL OF UNCERTAINTY. ON OUTPUT THESE PARAMETERS ARE */
/*         UPDATED APPROPRIATELY. */

/*       STP, FP, AND DP ARE VARIABLES WHICH SPECIFY THE STEP, */
/*         THE FUNCTION, AND THE DERIVATIVE AT THE CURRENT STEP. */
/*         IF BRACKT IS SET TRUE THEN ON INPUT STP MUST BE */
/*         BETWEEN STX AND STY. ON OUTPUT STP IS SET TO THE NEW STEP. */

/*       BRACKT IS A INT VARIABLE WHICH SPECIFIES IF A MINIMIZER */
/*         HAS BEEN BRACKETED. IF THE MINIMIZER HAS NOT BEEN BRACKETED */
/*         THEN ON INPUT BRACKT MUST BE SET FALSE. IF THE MINIMIZER */
/*         IS BRACKETED THEN ON OUTPUT BRACKT IS SET TRUE. */

/*       STPMIN AND STPMAX ARE INPUT VARIABLES WHICH SPECIFY LOWER */
/*         AND UPPER BOUNDS FOR THE STEP. */

/*       INFO IS AN INT OUTPUT VARIABLE SET AS FOLLOWS: */
/*         IF INFO = 1,2,3,4,5, THEN THE STEP HAS BEEN COMPUTED */
/*         ACCORDING TO ONE OF THE FIVE CASES BELOW. OTHERWISE */
/*         INFO = 0, AND THIS INDICATES IMPROPER INPUT PARAMETERS. */

/*     SUBPROGRAMS CALLED */

/*       FORTRAN-SUPPLIED ... ABS,MAX,MIN,SQRT */

/*     ARGONNE NATIONAL LABORATORY. MINPACK PROJECT. JUNE 1983 */
/*     JORGE J. MORE', DAVID J. THUENTE */

/*<       DOUBLE PRECISION GAMMA,P,Q,R,S,SGND,STPC,STPF,STPQ,THETA >*/
/*<       INFO = 0 >*/
    *info = 0;

/*     CHECK THE INPUT PARAMETERS FOR ERRORS. */

/*<        >*/
    if (*brackt && (*stp <= LBFGS_MIN(*stx,*sty) || *stp >= LBFGS_MAX(*stx,*sty)) || *dx *
	     (*stp - *stx) >= 0.f || *stpmax < *stpmin) {
	return 0;
    }

/*     DETERMINE IF THE DERIVATIVES HAVE OPPOSITE SIGN. */

/*<       SGND = DP*(DX/LBFGS_ABS(DX)) >*/
    sgnd = *dp * (*dx / LBFGS_ABS(*dx));

/*     FIRST CASE. A HIGHER FUNCTION VALUE. */
/*     THE MINIMUM IS BRACKETED. IF THE CUBIC STEP IS CLOSER */
/*     TO STX THAN THE QUADRATIC STEP, THE CUBIC STEP IS TAKEN, */
/*     ELSE THE AVERAGE OF THE CUBIC AND QUADRATIC STEPS IS TAKEN. */

/*<       IF (FP .GT. FX) THEN >*/
    if (*fp > *fx) {
/*<          INFO = 1 >*/
	*info = 1;
/*<          BOUND = .TRUE. >*/
    bound = LBFGS_TRUE;
/*<          THETA = 3*(FX - FP)/(STP - STX) + DX + DP >*/
	theta = (*fx - *fp) * 3 / (*stp - *stx) + *dx + *dp;
/*<          S = LBFGS_MAX(LBFGS_ABS(THETA),LBFGS_ABS(DX),LBFGS_ABS(DP)) >*/
/* Computing MAX */
	d__1 = LBFGS_ABS(theta), d__2 = LBFGS_ABS(*dx), d__1 = LBFGS_MAX(d__1,d__2), d__2 = LBFGS_ABS(
		*dp);
	s = LBFGS_MAX(d__1,d__2);
/*<          GAMMA = S*SQRT((THETA/S)**2 - (DX/S)*(DP/S)) >*/
/* Computing 2nd power */
	d__1 = theta / s;
	gamma = s * sqrt(d__1 * d__1 - *dx / s * (*dp / s));
/*<          IF (STP .LT. STX) GAMMA = -GAMMA >*/
	if (*stp < *stx) {
	    gamma = -gamma;
	}
/*<          P = (GAMMA - DX) + THETA >*/
	p = gamma - *dx + theta;
/*<          Q = ((GAMMA - DX) + GAMMA) + DP >*/
	q = gamma - *dx + gamma + *dp;
/*<          R = P/Q >*/
	r__ = p / q;
/*<          STPC = STX + R*(STP - STX) >*/
	stpc = *stx + r__ * (*stp - *stx);
/*<          STPQ = STX + ((DX/((FX-FP)/(STP-STX)+DX))/2)*(STP - STX) >*/
	stpq = *stx + *dx / ((*fx - *fp) / (*stp - *stx) + *dx) / 2 * (*stp - 
		*stx);
/*<          IF (LBFGS_ABS(STPC-STX) .LT. LBFGS_ABS(STPQ-STX)) THEN >*/
	if ((d__1 = stpc - *stx, LBFGS_ABS(d__1)) < (d__2 = stpq - *stx, LBFGS_ABS(d__2)))
		 {
/*<             STPF = STPC >*/
	    stpf = stpc;
/*<          ELSE >*/
	} else {
/*<            STPF = STPC + (STPQ - STPC)/2 >*/
	    stpf = stpc + (stpq - stpc) / 2;
/*<            END IF >*/
	}
/*<          BRACKT = .TRUE. >*/
    *brackt = LBFGS_TRUE;

/*     SECOND CASE. A LOWER FUNCTION VALUE AND DERIVATIVES OF */
/*     OPPOSITE SIGN. THE MINIMUM IS BRACKETED. IF THE CUBIC */
/*     STEP IS CLOSER TO STX THAN THE QUADRATIC (SECANT) STEP, */
/*     THE CUBIC STEP IS TAKEN, ELSE THE QUADRATIC STEP IS TAKEN. */

/*<       ELSE IF (SGND .LT. 0.0) THEN >*/
    } else if (sgnd < 0.f) {
/*<          INFO = 2 >*/
	*info = 2;
/*<          BOUND = .FALSE. >*/
    bound = LBFGS_FALSE;
/*<          THETA = 3*(FX - FP)/(STP - STX) + DX + DP >*/
	theta = (*fx - *fp) * 3 / (*stp - *stx) + *dx + *dp;
/*<          S = LBFGS_MAX(LBFGS_ABS(THETA),LBFGS_ABS(DX),LBFGS_ABS(DP)) >*/
/* Computing MAX */
	d__1 = LBFGS_ABS(theta), d__2 = LBFGS_ABS(*dx), d__1 = LBFGS_MAX(d__1,d__2), d__2 = LBFGS_ABS(
		*dp);
	s = LBFGS_MAX(d__1,d__2);
/*<          GAMMA = S*SQRT((THETA/S)**2 - (DX/S)*(DP/S)) >*/
/* Computing 2nd power */
	d__1 = theta / s;
	gamma = s * sqrt(d__1 * d__1 - *dx / s * (*dp / s));
/*<          IF (STP .GT. STX) GAMMA = -GAMMA >*/
	if (*stp > *stx) {
	    gamma = -gamma;
	}
/*<          P = (GAMMA - DP) + THETA >*/
	p = gamma - *dp + theta;
/*<          Q = ((GAMMA - DP) + GAMMA) + DX >*/
	q = gamma - *dp + gamma + *dx;
/*<          R = P/Q >*/
	r__ = p / q;
/*<          STPC = STP + R*(STX - STP) >*/
	stpc = *stp + r__ * (*stx - *stp);
/*<          STPQ = STP + (DP/(DP-DX))*(STX - STP) >*/
	stpq = *stp + *dp / (*dp - *dx) * (*stx - *stp);
/*<          IF (LBFGS_ABS(STPC-STP) .GT. LBFGS_ABS(STPQ-STP)) THEN >*/
	if ((d__1 = stpc - *stp, LBFGS_ABS(d__1)) > (d__2 = stpq - *stp, LBFGS_ABS(d__2)))
		 {
/*<             STPF = STPC >*/
	    stpf = stpc;
/*<          ELSE >*/
	} else {
/*<             STPF = STPQ >*/
	    stpf = stpq;
/*<             END IF >*/
	}
/*<          BRACKT = .TRUE. >*/
    *brackt = LBFGS_TRUE;

/*     THIRD CASE. A LOWER FUNCTION VALUE, DERIVATIVES OF THE */
/*     SAME SIGN, AND THE MAGNITUDE OF THE DERIVATIVE DECREASES. */
/*     THE CUBIC STEP IS ONLY USED IF THE CUBIC TENDS TO INFINITY */
/*     IN THE DIRECTION OF THE STEP OR IF THE MINIMUM OF THE CUBIC */
/*     IS BEYOND STP. OTHERWISE THE CUBIC STEP IS DEFINED TO BE */
/*     EITHER STPMIN OR STPMAX. THE QUADRATIC (SECANT) STEP IS ALSO */
/*     COMPUTED AND IF THE MINIMUM IS BRACKETED THEN THE THE STEP */
/*     CLOSEST TO STX IS TAKEN, ELSE THE STEP FARTHEST AWAY IS TAKEN. */

/*<       ELSE IF (LBFGS_ABS(DP) .LT. LBFGS_ABS(DX)) THEN >*/
    } else if (LBFGS_ABS(*dp) < LBFGS_ABS(*dx)) {
/*<          INFO = 3 >*/
	*info = 3;
/*<          BOUND = .TRUE. >*/
    bound = LBFGS_TRUE;
/*<          THETA = 3*(FX - FP)/(STP - STX) + DX + DP >*/
	theta = (*fx - *fp) * 3 / (*stp - *stx) + *dx + *dp;
/*<          S = LBFGS_MAX(LBFGS_ABS(THETA),LBFGS_ABS(DX),LBFGS_ABS(DP)) >*/
/* Computing MAX */
	d__1 = LBFGS_ABS(theta), d__2 = LBFGS_ABS(*dx), d__1 = LBFGS_MAX(d__1,d__2), d__2 = LBFGS_ABS(
		*dp);
	s = LBFGS_MAX(d__1,d__2);

/*        THE CASE GAMMA = 0 ONLY ARISES IF THE CUBIC DOES NOT TEND */
/*        TO INFINITY IN THE DIRECTION OF THE STEP. */

/*<          GAMMA = S*SQRT(LBFGS_MAX(0.0D0,(THETA/S)**2 - (DX/S)*(DP/S))) >*/
/* Computing MAX */
/* Computing 2nd power */
	d__3 = theta / s;
	d__1 = 0., d__2 = d__3 * d__3 - *dx / s * (*dp / s);
	gamma = s * sqrt((LBFGS_MAX(d__1,d__2)));
/*<          IF (STP .GT. STX) GAMMA = -GAMMA >*/
	if (*stp > *stx) {
	    gamma = -gamma;
	}
/*<          P = (GAMMA - DP) + THETA >*/
	p = gamma - *dp + theta;
/*<          Q = (GAMMA + (DX - DP)) + GAMMA >*/
	q = gamma + (*dx - *dp) + gamma;
/*<          R = P/Q >*/
	r__ = p / q;
/*<          IF (R .LT. 0.0 .AND. GAMMA .NE. 0.0) THEN >*/
	if (r__ < 0.f && gamma != 0.f) {
/*<             STPC = STP + R*(STX - STP) >*/
	    stpc = *stp + r__ * (*stx - *stp);
/*<          ELSE IF (STP .GT. STX) THEN >*/
	} else if (*stp > *stx) {
/*<             STPC = STPMAX >*/
	    stpc = *stpmax;
/*<          ELSE >*/
	} else {
/*<             STPC = STPMIN >*/
	    stpc = *stpmin;
/*<             END IF >*/
	}
/*<          STPQ = STP + (DP/(DP-DX))*(STX - STP) >*/
	stpq = *stp + *dp / (*dp - *dx) * (*stx - *stp);
/*<          IF (BRACKT) THEN >*/
	if (*brackt) {
/*<             IF (LBFGS_ABS(STP-STPC) .LT. LBFGS_ABS(STP-STPQ)) THEN >*/
	    if ((d__1 = *stp - stpc, LBFGS_ABS(d__1)) < (d__2 = *stp - stpq, LBFGS_ABS(
		    d__2))) {
/*<                STPF = STPC >*/
		stpf = stpc;
/*<             ELSE >*/
	    } else {
/*<                STPF = STPQ >*/
		stpf = stpq;
/*<                END IF >*/
	    }
/*<          ELSE >*/
	} else {
/*<             IF (LBFGS_ABS(STP-STPC) .GT. LBFGS_ABS(STP-STPQ)) THEN >*/
	    if ((d__1 = *stp - stpc, LBFGS_ABS(d__1)) > (d__2 = *stp - stpq, LBFGS_ABS(
		    d__2))) {
/*<                STPF = STPC >*/
		stpf = stpc;
/*<             ELSE >*/
	    } else {
/*<                STPF = STPQ >*/
		stpf = stpq;
/*<                END IF >*/
	    }
/*<             END IF >*/
	}

/*     FOURTH CASE. A LOWER FUNCTION VALUE, DERIVATIVES OF THE */
/*     SAME SIGN, AND THE MAGNITUDE OF THE DERIVATIVE DOES */
/*     NOT DECREASE. IF THE MINIMUM IS NOT BRACKETED, THE STEP */
/*     IS EITHER STPMIN OR STPMAX, ELSE THE CUBIC STEP IS TAKEN. */

/*<       ELSE >*/
    } else {
/*<          INFO = 4 >*/
	*info = 4;
/*<          BOUND = .FALSE. >*/
    bound = LBFGS_FALSE;
/*<          IF (BRACKT) THEN >*/
	if (*brackt) {
/*<             THETA = 3*(FP - FY)/(STY - STP) + DY + DP >*/
	    theta = (*fp - *fy) * 3 / (*sty - *stp) + *dy + *dp;
/*<             S = LBFGS_MAX(LBFGS_ABS(THETA),LBFGS_ABS(DY),LBFGS_ABS(DP)) >*/
/* Computing MAX */
	    d__1 = LBFGS_ABS(theta), d__2 = LBFGS_ABS(*dy), d__1 = LBFGS_MAX(d__1,d__2), d__2 = 
		    LBFGS_ABS(*dp);
	    s = LBFGS_MAX(d__1,d__2);
/*<             GAMMA = S*SQRT((THETA/S)**2 - (DY/S)*(DP/S)) >*/
/* Computing 2nd power */
	    d__1 = theta / s;
	    gamma = s * sqrt(d__1 * d__1 - *dy / s * (*dp / s));
/*<             IF (STP .GT. STY) GAMMA = -GAMMA >*/
	    if (*stp > *sty) {
		gamma = -gamma;
	    }
/*<             P = (GAMMA - DP) + THETA >*/
	    p = gamma - *dp + theta;
/*<             Q = ((GAMMA - DP) + GAMMA) + DY >*/
	    q = gamma - *dp + gamma + *dy;
/*<             R = P/Q >*/
	    r__ = p / q;
/*<             STPC = STP + R*(STY - STP) >*/
	    stpc = *stp + r__ * (*sty - *stp);
/*<             STPF = STPC >*/
	    stpf = stpc;
/*<          ELSE IF (STP .GT. STX) THEN >*/
	} else if (*stp > *stx) {
/*<             STPF = STPMAX >*/
	    stpf = *stpmax;
/*<          ELSE >*/
	} else {
/*<             STPF = STPMIN >*/
	    stpf = *stpmin;
/*<             END IF >*/
	}
/*<          END IF >*/
    }

/*     UPDATE THE INTERVAL OF UNCERTAINTY. THIS UPDATE DOES NOT */
/*     DEPEND ON THE NEW STEP OR THE CASE ANALYSIS ABOVE. */

/*<       IF (FP .GT. FX) THEN >*/
    if (*fp > *fx) {
/*<          STY = STP >*/
	*sty = *stp;
/*<          FY = FP >*/
	*fy = *fp;
/*<          DY = DP >*/
	*dy = *dp;
/*<       ELSE >*/
    } else {
/*<          IF (SGND .LT. 0.0) THEN >*/
	if (sgnd < 0.f) {
/*<             STY = STX >*/
	    *sty = *stx;
/*<             FY = FX >*/
	    *fy = *fx;
/*<             DY = DX >*/
	    *dy = *dx;
/*<             END IF >*/
	}
/*<          STX = STP >*/
	*stx = *stp;
/*<          FX = FP >*/
	*fx = *fp;
/*<          DX = DP >*/
	*dx = *dp;
/*<          END IF >*/
    }

/*     COMPUTE THE NEW STEP AND SAFEGUARD IT. */

/*<       STPF = LBFGS_MIN(STPMAX,STPF) >*/
    stpf = LBFGS_MIN(*stpmax,stpf);
/*<       STPF = LBFGS_MAX(STPMIN,STPF) >*/
    stpf = LBFGS_MAX(*stpmin,stpf);
/*<       STP = STPF >*/
    *stp = stpf;
/*<       IF (BRACKT .AND. BOUND) THEN >*/
    if (*brackt && bound) {
/*<          IF (STY .GT. STX) THEN >*/
	if (*sty > *stx) {
/*<             STP = LBFGS_MIN(STX+0.66*(STY-STX),STP) >*/
/* Computing MIN */
	    d__1 = *stx + (*sty - *stx) * .66f;
	    *stp = LBFGS_MIN(d__1,*stp);
/*<          ELSE >*/
	} else {
/*<             STP = LBFGS_MAX(STX+0.66*(STY-STX),STP) >*/
/* Computing MAX */
	    d__1 = *stx + (*sty - *stx) * .66f;
	    *stp = LBFGS_MAX(d__1,*stp);
/*<             END IF >*/
	}
/*<          END IF >*/
    }
/*<       RETURN >*/
    return 0;

/*     LAST LINE OF SUBROUTINE MCSTEP. */

/*<       END >*/
} /* mcstep_ */

