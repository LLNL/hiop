#ifndef HIOP_EXAMPLE_EX6
#define  HIOP_EXAMPLE_EX6

#include "hiopInterface.hpp"

#include <cassert>

#ifdef HIOP_USE_MPI
#include "mpi.h"
#else
#define MPI_COMM_WORLD 0
#define MPI_Comm int
#endif

using int_type = hiop::int_type;

/* Test with bounds and constraints of all types. For some reason this
 *  example is not very well behaved numerically.
 *  min   sum 1/4* { (x_{i}-1)^4 : i=1,...,n}
 *  s.t.
 *            4*x_1 + 2*x_2                     == 10
 *        5<= 2*x_1         + x_3
 *        1<= 2*x_1                 + 0.5*x_4   <= 2*n, for i=4,...,n
 *        x_1 free
 *        0.0 <= x_2
 *        1.5 <= x_3 <= 10
 *        x_i >=0.5, i=4,...,n
 */
class Ex6 : public hiop::hiopInterfaceSparse
{
public:
  Ex6(int n, double scal_in);
  virtual ~Ex6();

  virtual bool get_prob_sizes(int_type& n, int_type& m);
  virtual bool get_vars_info(const int_type& n, double *xlow, double* xupp, NonlinearityType* type);
  virtual bool get_cons_info(const int_type& m, double* clow, double* cupp, NonlinearityType* type);
  virtual bool get_sparse_blocks_info(int& nx,
					    int& nnz_sparse_Jaceq, int& nnz_sparse_Jacineq,
					    int& nnz_sparse_Hess_Lagr);

  virtual bool eval_f(const int_type& n, const double* x, bool new_x, double& obj_value);
  virtual bool eval_cons(const int_type& n, const int_type& m,
			 const int_type& num_cons, const int_type* idx_cons,
			 const double* x, bool new_x, double* cons);
  virtual bool eval_cons(const int_type& n, const int_type& m,
			 const double* x, bool new_x,
			 double* cons);
  virtual bool eval_grad_f(const int_type& n, const double* x, bool new_x, double* gradf);
  virtual bool eval_Jac_cons(const int_type& n, const int_type& m,
			     const int_type& num_cons, const int_type* idx_cons,
			     const double* x, bool new_x,
			     const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS);
  virtual bool eval_Jac_cons(const int_type& n, const int_type& m,
			     const double* x, bool new_x,
			     const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS);
  virtual bool get_starting_point(const int_type&n, double* x0);
  virtual bool eval_Hess_Lagr(const int_type& n, const int_type& m,
			      const double* x, bool new_x, const double& obj_factor,
			      const double* lambda, bool new_lambda,
			      const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS);

private:
  int n_vars, n_cons;
  double scal;
};
#endif
