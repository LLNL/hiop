#ifndef HIOP_EXAMPLE_EX2
#define  HIOP_EXAMPLE_EX2

#include "hiopInterface.hpp"

#include <cassert>

#ifdef HIOP_USE_MPI
#include "mpi.h"
#else
#define MPI_COMM_WORLD 0
#define MPI_Comm int
#endif

/* Test with bounds and constraints of all types. For some reason this
 *  example is not very well behaved numerically.
 *  min   sum 1/4* { (x_{i}-1)^4 : i=1,...,n}
 *  s.t.  
 *        sum x_i = n+1
 *        5<= 2*x_1 + sum {x_i : i=2,...,n} 
 *        1<= 2*x_1 + 0.5*x_2 + sum{x_i : i=3,...,n} <= 2*n
 *            4*x_1 + 2  *x_2 + 2*x_3 + sum{x_i : i=4,...,n} <=4*n
 *        x_1 free
 *        0.0 <= x_2 
 *        1.5 <= x_3 <= 10
 *        x_i >=0.5, i=4,...,n
 */
class Ex2 : public hiop::hiopInterfaceDenseConstraints
{
public: 
  Ex2(int n);
  virtual ~Ex2();

  virtual bool get_prob_sizes(long long& n, long long& m);
  virtual bool get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type);
  virtual bool get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type);

  virtual bool eval_f(const long long& n, const double* x, bool new_x, double& obj_value);
  virtual bool eval_cons(const long long& n, const long long& m, 
			 const long long& num_cons, const long long* idx_cons,  
			 const double* x, bool new_x, double* cons);
  virtual bool eval_grad_f(const long long& n, const double* x, bool new_x, double* gradf);
  virtual bool eval_Jac_cons(const long long& n, const long long& m,
			     const long long& num_cons, const long long* idx_cons,  
			     const double* x, bool new_x, double** Jac);
  virtual bool get_vecdistrib_info(long long global_n, long long* cols);

  virtual bool get_starting_point(const long long&n, double* x0);

  /*
    void solution_callback(hiop::hiopSolveStatus status,
			 int n, const double* x,
			 const double* z_L,
			 const double* z_U,
			 int m, const double* g,
			 const double* lambda,
			 double obj_value) { 
    printf("solution_callback with optimal value: %g. Also x[1]=%22.14f\n", obj_value, x[1]);
};
  

    virtual bool iterate_callback(int iter, double obj_value,
				int n, const double* x,
				const double* z_L,
				const double* z_U,
				int m, const double* g,
				const double* lambda,
				double inf_pr, double inf_du,
				double mu,
				double alpha_du, double alpha_pr,
  				int ls_trials) {if(iter==3)return false;printf("%g %g\n", x[0], x[1]); return true;}
  */
private:
  int n_vars, n_cons;
  MPI_Comm comm;
  int my_rank, comm_size;
  long long* col_partition;
public:
  inline long long idx_local2global(long long global_n, int idx_local) 
  { 
    assert(idx_local + col_partition[my_rank]<col_partition[my_rank+1]);
    if(global_n==n_vars)
      return idx_local + col_partition[my_rank]; 
    assert(false && "you shouldn't need global index for a vector of this size.");
    return -1;
  }
  inline int idx_global2local(long long global_n, long long idx_global)
  {
    assert(idx_global>=col_partition[my_rank]   && "global index does not belong to this rank");
    assert(idx_global< col_partition[my_rank+1] && "global index does not belong to this rank");
    assert(global_n==n_vars && "your global_n does not match the number of variables?");
    return (int) (idx_global-col_partition[my_rank]);
  }
};
#endif
