#ifndef HIOP_EXAMPLE_RAJA_EX4
#define HIOP_EXAMPLE_RAJA_EX4

#include "hiopInterface.hpp"

//this include is not needed in general
//we use hiopMatrixDense in this particular example for convienience
#include <hiopMatrixDense.hpp>
#include <hiopLinAlgFactory.hpp>

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

/* Problem test for the linear algebra of Mixed Dense-Sparse NLPs
 *  min   sum 0.5 {x_i*(x_{i}-1) : i=1,...,ns} + 0.5 y'*Qd*y + 0.5 s^T s
 *  s.t.  x+s + Md y = 0, i=1,...,ns
 *        [-2  ]    [ x_1 + e^T s]   [e^T]      [ 2 ]
 *        [-inf] <= [ x_2        ] + [e^T] y <= [ 2 ]
 *        [-2  ]    [ x_3        ]   [e^T]      [inf]
 *        x <= 3
 *        s>=0
 *        -4 <=y_1 <=4, the rest of y are free
 *        
 * The vector 'y' is of dimension nd = ns (can be changed in the constructor)
 * Dense matrices Qd and Md are such that
 * Qd  = two on the diagonal, one on the first offdiagonals, zero elsewhere
 * Md  = minus one everywhere
 * e   = vector of all ones
 *
 * Coding of the problem in MDS HiOp input: order of variables need to be [x,s,y] 
 * since [x,s] are the so-called sparse variables and y are the dense variables
 */

class Ex4 : public hiop::hiopInterfaceMDS
{
public:
  Ex4(int ns_in, std::string mem_space)
    : Ex4(ns_in, ns_in, mem_space)
  {
  }
  
  Ex4(int ns_in, int nd_in, std::string mem_space);

  virtual ~Ex4();
  
  bool get_prob_sizes(long long& n, long long& m);

  /**
   * @todo will param _type_ live on host or device?
   * @todo register pointers with umpire in case they need to be copied
   * from device to host.
   */
  bool get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type);

  /**
   * @todo fill out param descriptions below to determine whether or not
   * they will reside on device and will have to be accessed/assigned to 
   * in a RAJA kernel
   *
   * @param[out] m ?
   * @param[out] clow ?
   * @param[out] cupp ?
   * @param[out] type ?
   */
  bool get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type);

  bool get_sparse_dense_blocks_info(int& nx_sparse, int& nx_dense,
				    int& nnz_sparse_Jace, int& nnz_sparse_Jaci,
				    int& nnz_sparse_Hess_Lagr_SS, int& nnz_sparse_Hess_Lagr_SD);
  
  bool eval_f(const long long& n, const double* x, bool new_x, double& obj_value);

  /**
   * @todo figure out which of these pointers (if any) will need to be
   * copied over to device when this is fully running on device.
   * @todo find descriptoins of parameters (perhaps from ipopt interface?).
   *
   * @param[in] idx_cons ?
   * @param[in] x ?
   * @param[in] cons ?
   */
  virtual bool eval_cons(const long long& n, const long long& m, 
			 const long long& num_cons, const long long* idx_cons,  
			 const double* x, bool new_x, double* cons);

  bool eval_cons(const long long& n, const long long& m, 
      const double* x, bool new_x, double* cons)
  {
    //return false so that HiOp will rely on the constraint evaluator defined above
    return false;
  }

  //sum 0.5 {x_i*(x_{i}-1) : i=1,...,ns} + 0.5 y'*Qd*y + 0.5 s^T s
  bool eval_grad_f(const long long& n, const double* x, bool new_x, double* gradf);

  /**
   * @todo This method will probably always have to run on the CPU side since
   * the var _nnzit_ is loop-dependent and cannot be run in parallel with the
   * other loop iterations.
   */
  virtual bool eval_Jac_cons(const long long& n, const long long& m, 
        const long long& num_cons, const long long* idx_cons,
        const double* x, bool new_x,
        const long long& nsparse, const long long& ndense, 
        const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
        double** JacD);

  virtual bool eval_Jac_cons(const long long& n, const long long& m, 
        const double* x, bool new_x,
        const long long& nsparse, const long long& ndense, 
        const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
        double** JacD)
  {
    //return false so that HiOp will rely on the Jacobian evaluator defined above
    return false;
  }

  bool eval_Hess_Lagr(const long long& n, const long long& m, 
      const double* x, bool new_x, const double& obj_factor,
      const double* lambda, bool new_lambda,
      const long long& nsparse, const long long& ndense, 
      const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS, 
      double** HDD,
      int& nnzHSD, int* iHSD, int* jHSD, double* MHSD);

  /* Implementation of the primal starting point specification */
  bool get_starting_point(const long long& global_n, double* x0);

  bool get_starting_point(const long long& n, const long long& m,
      double* x0,
      bool& duals_avail,
      double* z_bndL0, double* z_bndU0,
      double* lambda0);

  /* The public methods below are not part of hiopInterface. They are a proxy
   * for user's (front end) code to set solutions from a previous solve. 
   *
   * Same behaviour can be achieved internally (in this class ) if desired by 
   * overriding @solution_callback and @get_starting_point
   */
  void set_solution_primal(const double* x_vec);

  void set_solution_duals(const double* zl_vec, const double* zu_vec, const double* lambda_vec);

  void initialize();
  
protected:
  int ns_, nd_;
  hiop::hiopMatrixDense* Q_;
  hiop::hiopMatrixDense* Md_;
  double* buf_y_;
  bool haveIneq_;
  std::string mem_space_;

  /* Internal buffers to store primal-dual solution */
  double* sol_x_;
  double* sol_zl_;
  double* sol_zu_;
  double* sol_lambda_;
};

class Ex4OneCallCons : public Ex4
{
  public:
    Ex4OneCallCons(int ns_in, std::string mem_space)
      : Ex4(ns_in, mem_space)
    {
    }

    Ex4OneCallCons(int ns_in, int nd_in, std::string mem_space)
      : Ex4(ns_in, nd_in, mem_space)
    {
    }

    virtual ~Ex4OneCallCons()
    {
    }

    bool eval_cons(const long long& n, const long long& m, 
        const long long& num_cons, const long long* idx_cons,  
        const double* x, bool new_x, double* cons)
    {
      //return false so that HiOp will rely on the on-call constraint evaluator defined below
      return false;
    }

    /** all constraints evaluated in here */
    bool eval_cons(const long long& n, const long long& m, 
        const double* x, bool new_x, double* cons);

    virtual bool
      eval_Jac_cons(const long long& n, const long long& m, 
          const long long& num_cons, const long long* idx_cons,
          const double* x, bool new_x,
          const long long& nsparse, const long long& ndense, 
          const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
          double** JacD)
      {
        return false; // so that HiOp will call the one-call full-Jacob function below
      }

    virtual bool eval_Jac_cons(const long long& n, const long long& m, 
          const double* x, bool new_x,
          const long long& nsparse, const long long& ndense, 
          const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
          double** JacD);
};

#endif
