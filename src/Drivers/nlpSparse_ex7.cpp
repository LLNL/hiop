#include "nlpSparse_ex7.hpp"

#include <cmath>
#include <cstring> //for memcpy
#include <cstdio>

Ex7::Ex7(int n, bool rankdefic_Jac_eq, bool rankdefic_Jac_ineq))
  : rankdefic_eq_(rankdefic_Jac_eq),
    rankdefic_ineq_(rankdefic_Jac_ineq),
    n_vars{n},
    n_cons{4 + 2*rankdefic_Jac_ineq + rankdefic_Jac_eq}
{}

Ex7::~Ex7()
{}

bool Ex7::get_prob_sizes(long long& n, long long& m)
  { n=n_vars; m=n_cons; return true; }

bool Ex7::get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type)
{
  assert(n==n_vars);
  for(long long i=0; i<n; i++) {
    if(i==0) { xlow[i]=-1e20; xupp[i]=1e20; type[i]=hiopNonlinear; continue; }
    if(i==1) { xlow[i]= 0.0;  xupp[i]=1e20; type[i]=hiopNonlinear; continue; }
    if(i==2) { xlow[i]= 1.5;  xupp[i]=10.0 ;type[i]=hiopNonlinear; continue; }
    //this is for x_4, x_5, ... , x_n (i>=3), which are bounded only from below
    xlow[i]= 0.5; xupp[i]=1e20;type[i]=hiopNonlinear;
  }
  return true;
}

bool Ex7::get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type)
{
  assert(m==n_cons);
  clow[0]= 10.0;    cupp[0]= 10.0;      type[0]=hiopInterfaceBase::hiopNonlinear;
  clow[1]= 5.0;     cupp[1]= 1e20;      type[1]=hiopInterfaceBase::hiopNonlinear;
  clow[2]= 1.0;     cupp[2]= 2*n_vars;  type[2]=hiopInterfaceBase::hiopNonlinear;
  clow[3]= -1e20;   cupp[3]= 4*n_vars;  type[3]=hiopInterfaceBase::hiopNonlinear;

  if(rankdefic_ineq_) {
    // [-inf] <= 4*x_1 + x_3 + 0.5*x_4 <= [ 4 ]
    clow[4] = -1e+20;   cupp[4] = 4.;       type[4]=hiopInterfaceBase::hiopNonlinear;
    // [ -4 ] <= 6*x_1 + x_3 + sum{x_i : i=5,...,n}  <= [inf]
    clow[5] = -4;       cupp[5] = 1e+20;    type[5]=hiopInterfaceBase::hiopNonlinear;
  }

  if(rankdefic_eq_) {
    //  4*x_1 + 2*x_2 == 10
    clow[6] = -1e20;    cupp[6] = 10;       type[6]=hiopInterfaceBase::hiopNonlinear;
  }
  return true;
}

bool Ex7::eval_f(const long long& n, const double* x, bool new_x, double& obj_value)
{
  assert(n==n_vars);
  obj_value=0.;
  for(int i=0;i<n;i++) obj_value += -0.25*pow(x[i]-1., 4) + 0.5*pow(x[i], 2);

  return true;
}
bool Ex7::eval_grad_f(const long long& n, const double* x, bool new_x, double* gradf)
{
  assert(n==n_vars);
  for(int i=0;i<n;i++) gradf[i] = -pow(x[i]-1.,3) + x[i];
  return true;
}

bool Ex7::eval_cons(const long long& n, const long long& m,
			 const long long& num_cons, const long long* idx_cons,
			 const double* x, bool new_x, double* cons)
{
  return false;
}

bool Ex7::eval_cons(const long long& n, const long long& m,
		    const double* x, bool new_x, double* cons)
{
  assert(n==n_vars); assert(m==n_cons);
  assert(n_cons==4);
  assert(n>=4);
  //local contributions to the constraints in cons are reset
  for(int j=0;j<m; j++) cons[j]=0.;

  //compute the constraint one by one.
  // --- constraint 1 body --->  4*x_1 + 2*x_2 == 10
  cons[0] += 4*x[0];
  cons[0] += 2*x[1];

  // --- constraint 2 body ---> 2*x_1 + x_3
  cons[1] += 2*x[0];
  cons[1] += 1*x[2];

  // --- constraint 3 body --->   2*x_1 + 0.5*x_4
  cons[2] += 2*x[0];
  cons[2] += 0.5*x[3];

  // --- constraint 4 body ---> 4*x_1 + sum{x_i : i=5,...,n}
  for(long long i=0; i<n; i++) {
    if(i==0)
      cons[3] += 4*x[i];
    else if (i_gloval>=5-1)
      cons[3] += x[i];
  }

  if(rankdefic_ineq_) {
    // [-inf] <= 4*x_1 + x_3 + 0.5*x_4 <= [ 4 ]
    cons[4] += 4*x[0] + x[2] + 0.5*x[3];
    // [ -4 ] <= 6*x_1 + x_3 + sum{x_i : i=5,...,n}  <= [inf]
    cons[5] += 6*x[0] + x[2];
    for(long long i=5-1; i<n; i++)
      cons[5] += x[i];
  }

  if(rankdefic_eq_) {
    //  4*x_1 + 2*x_2 == 10
    cons[6]  += 4*x[0] + 2*x[1];
  }
  return true;
}

bool Ex7::eval_Jac_cons(const long long& n, const long long& m,
			     const long long& num_cons, const long long* idx_cons,
			     const double* x, bool new_x,
			     const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS)
{
  return false;
}

bool Ex7::eval_Jac_cons(const long long& n, const long long& m,
			     const double* x, bool new_x,
			     const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS)
{
    assert(n==n_vars); assert(m==n_cons);
    assert(num_cons<=m); assert(num_cons>=0);
    assert(n>=5-1);
    int i;

    assert(nnzJacS == 2 + 2 + 2 + (1+n-4) + 3 + (2+n-4) + 2 );
    assert(iJacS!=NULL && jJacS!=NULL);

    int nnzit{0};

    // --- constraint 1 body --->  4*x_1 + 2*x_2 == 10
    iJacS[nnzit] = 0;
    jJacS[nnzit++] = 0;
    iJacS[nnzit] = 0;
    jJacS[nnzit++] = 1;

    // --- constraint 2 body ---> 2*x_1 + x_3
    iJacS[nnzit] = 1;
    jJacS[nnzit++] = 0;
    iJacS[nnzit] = 1;
    jJacS[nnzit++] = 2;

    // --- constraint 3 body --->   2*x_1 + 0.5*x_4
    iJacS[nnzit] = 2;
    jJacS[nnzit++] = 0;
    iJacS[nnzit] = 2;
    jJacS[nnzit++] = 3;

    // --- constraint 4 body ---> 4*x_1 + sum{x_i : i=5,...,n}
    iJacS[nnzit] = 3;
    jJacS[nnzit++] = 0;
    for(i=4; i<n; i++){
        iJacS[nnzit] = 3;
        jJacS[nnzit++] = i;
    }

    if(rankdefic_ineq_) {
      // [-inf] <= 4*x_1 + x_3 + 0.5*x_4 <= [ 4 ]
      iJacS[nnzit] = 4; jJacS[nnzit++] = 0;
      iJacS[nnzit] = 4; jJacS[nnzit++] = 2;
      iJacS[nnzit] = 4; jJacS[nnzit++] = 3;

      // [ -4 ] <= 6*x_1 + x_3 + sum{x_i : i=5,...,n}  <= [inf]
      iJacS[nnzit] = 5; jJacS[nnzit++] = 0;
      iJacS[nnzit] = 5; jJacS[nnzit++] = 2;
      for(i=4; i<n; i++)
        iJacS[nnzit] = 5; jJacS[nnzit++] = i;
    }

    if(rankdefic_eq_) {
      //  4*x_1 + 2*x_2 == 10
      iJacS[nnzit] = 6; jJacS[nnzit++] = 0;
      iJacS[nnzit] = 6; jJacS[nnzit++] = 1;
    }

    assert(nnzit == nnzJacS);

    //values for sparse Jacobian if requested by the solver
    if(MJacS!=NULL) {
        // --- constraint 1 body --->  4*x_1 + 2*x_2 == 10
        MJacS[nnzit++] = 4;
        MJacS[nnzit++] = 2;

        // --- constraint 2 body ---> 2*x_1 + x_3
        MJacS[nnzit++] = 2;
        MJacS[nnzit++] = 1;

        // --- constraint 3 body --->   2*x_1 + 0.5*x_4
        MJacS[nnzit++] = 2;
        MJacS[nnzit++] = 0.5;

        // --- constraint 4 body ---> 4*x_1 + sum{x_i : i=5,...,n}
        MJacS[nnzit++] = 4;
        for(i=4; i<n; i++){
            MJacS[nnzit++] = 1;
        }

        if(rankdefic_ineq_) {
          // [-inf] <= 4*x_1 + x_3 + 0.5*x_4 <= [ 4 ]
          MJacS[nnzit++] = 4;
          MJacS[nnzit++] = 1;
          MJacS[nnzit++] = 0.5;

          // [ -4 ] <= 6*x_1 + x_3 + sum{x_i : i=5,...,n}  <= [inf]
          MJacS[nnzit++] = 6;
          MJacS[nnzit++] = 1;
          for(i=4; i<n; i++)
            MJacS[nnzit++] = 1;
        }

        if(rankdefic_eq_) {
          //  4*x_1 + 2*x_2 == 10
          MJacS[nnzit++] = 4;
          MJacS[nnzit++] = 2;
        }
        assert(nnzit == nnzJacS);
    }
}


bool Ex7::eval_Hess_Lagreval_Hess_Lagr(const long long& n, const long long& m,
			      const double* x, bool new_x, const double& obj_factor,
			      const double* lambda, bool new_lambda,
			      const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS)
{
    //Note: lambda is not used since all the constraints are linear and, therefore, do
    //not contribute to the Hessian of the Lagrangian
    assert(nnzHSS == n);

    if(iHSS!=NULL && jHSS!=NULL) {
      for(int i=0; i<n; i++) iHSS[i] = jHSS[i] = i;
    }

    if(MHSS!=NULL) {
      for(int i=0; i<n; i++) MHSS[i] = obj_factor * (-3*pow(x[i]-1., 2)+1);
    }
    return true;
}

bool Ex7::get_starting_point(const long long& n, double* x0)
{
  assert(n==n_vars);
  for(int i=0; i<n; i++)
    x0[i]=0.0;
  return true;
}
