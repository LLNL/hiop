#ifndef HIOP_EXAMPLE_EX7
#define HIOP_EXAMPLE_EX7

#include "hiopInterface.hpp"

//this include is not needed in general
//we use hiopMatrixSparse in this particular example for convenience
#include "hiopMatrixSparse.hpp"
#include "hiopLinAlgFactory.hpp"

#ifdef HIOP_USE_MPI
#include "mpi.h"
#else
#define MPI_COMM_WORLD 0
#define MPI_Comm int
#endif

#include <cassert>
#include <cstring> //for memcpy
#include <cstdio>
#include <cmath>

/** Nonlinear *highly nonconvex* and *rank deficient* problem test for the Filter IPM
 * Newton of HiOp. It uses a Sparse NLP formulation. The problem is based on Ex6.
 *
 *  min   -(2*convex_obj-1)*sum 1/4* { (x_{i}-1)^4 : i=1,...,n} + 0.5x^Tx
 *  s.t.
 *            4*x_1 + 2*x_2                     == 10
 *        5<= 2*x_1         + x_3
 *        1<= 2*x_1                 + 0.5*x_i   <= 2*n, for i=4,...,n
 *        x_1 free
 *        0.0 <= x_2
 *        1.5 <= x_3 <= 10
 *        x_i >=0.5, i=4,...,n
 *
 * Optionally, one can add the following constraints to obtain a rank-deficient Jacobian
 *
 *  s.t.  [-inf] <= 4*x_1 + 2*x_3 <= [ 19 ]                  (rnkdef-con1)
 *        4*x_1 + 2*x_2 == 10                                (rnkdef-con2)
 *
 *
 */
class Ex7 : public hiop::hiopInterfaceSparse
{
public:
  Ex7(int n, bool convex_obj, bool rankdefic_Jac_eq, bool rankdefic_Jac_ineq);
  virtual ~Ex7();

  virtual bool get_prob_sizes(long long& n, long long& m);
  virtual bool get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type);
  virtual bool get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type);
  virtual bool get_sparse_blocks_info(int& nx,
					    int& nnz_sparse_Jaceq, int& nnz_sparse_Jacineq,
					    int& nnz_sparse_Hess_Lagr);

  virtual bool eval_f(const long long& n, const double* x, bool new_x, double& obj_value);
  virtual bool eval_cons(const long long& n, const long long& m,
			 const long long& num_cons, const long long* idx_cons,
			 const double* x, bool new_x, double* cons);
  virtual bool eval_cons(const long long& n, const long long& m,
			 const double* x, bool new_x,
			 double* cons);
  virtual bool eval_grad_f(const long long& n, const double* x, bool new_x, double* gradf);
  virtual bool eval_Jac_cons(const long long& n, const long long& m,
			     const long long& num_cons, const long long* idx_cons,
			     const double* x, bool new_x,
			     const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS);
  virtual bool eval_Jac_cons(const long long& n, const long long& m,
			     const double* x, bool new_x,
			     const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS);
  virtual bool get_starting_point(const long long&n, double* x0);
  virtual bool eval_Hess_Lagr(const long long& n, const long long& m,
			      const double* x, bool new_x, const double& obj_factor,
			      const double* lambda, bool new_lambda,
			      const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS);

private:
  int n_vars, n_cons;
  bool convex_obj_, rankdefic_eq_, rankdefic_ineq_;
};

#endif
