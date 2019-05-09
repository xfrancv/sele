/*--------------------------------------------------------------
  Example of using LBFGS for minimizing a simple function.

  (C) 2014 V.Franc
----------------------------------------------------------------*/

#include <iostream>
#include "lbfgs.h"

using namespace std;

// Derived class for minimization of a benchmark function
// from original Nocedal's code. Optimal function value is zero,
// optimal solution vector is 1,1,...1.
class MySolver : public CLBFGS {

public:
    MySolver() : CLBFGS () {}

    // evulate function value and gradient at x
    bool EvalFunction( double* x, double &fval, double* grad )
    {
        fval = 0.;
        for (int j = 0; j < n_; j += 2)
        {
          double t1 = 1. - x[j];
          double t2 = (x[j+1] - x[j]* x[j]) * 10.;

          grad[j+1] = t2 * 20.;
          grad[j] = (x[j] * grad[j+1] + t1) * -2.;

          fval = fval + t1*t1 + t2*t2;
        }

        return true;
    }
};

int main()
{
    MySolver sol;

    int max_num_iter = 100;
    int n = 100;
    double* x0 = new double[n];

    for(int j = 0; j < n; j += 2)
    {
       x0[j] = -10.2;
       x0[j+1] = 10.2;
    }

    sol.Solve( n, x0, max_num_iter );

    cout << "function value: " << sol.fval() << endl;
    cout << "gradient norm: " << sol.gnorm() << endl;
    cout << "solution vector norm: " << sol.xnorm() << endl;
    cout << "solution vector:" << endl;
    for(int i = 0; i < n; i++ ) cout << i << ":" << sol.x(i) << " ";
    cout << endl;

    return 0;
}

