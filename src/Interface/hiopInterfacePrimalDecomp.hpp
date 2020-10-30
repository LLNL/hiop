#ifndef HIOP_INTERFACE_PRIDEC
#define HIOP_INTERFACE_PRIDEC

//for solve status
#include "hiopInterface.hpp"

namespace hiop
{
  
/**
 * Base class (interface) for specifying optimization NLPs that have separable terms in the 
 * objective, which we coin as "primal decomposable" problems. 
 * 
 * The following structure and terminology is assumed
 *
 *   min  basecase + sum { r_i(x) : i=1,...,S}      (primal decomposable NLP)
 *    x
 *
 * where 'basecase' refers to an NLP of a general form in the optimization variable 'x'. 
 * Furthermore, borrowing from stochastic programming terminology, the terms 'r_i' are called 
 * recourse terms, or, in short, r-term.
 * 
 * In order to solve the above problem, HiOp solver will perform a series of approximations and will
 * require the user to solve a so-called 'master' problem
 * 
 *   min  basecase +  q(x)                            (master NLP)
 *    x
 * where q(x) are convex differentiable approximations of r_i(x). 
 * 
 * The user is required to maintain and solve the master problem, more specifically
 *  - to add quadratic regularization to the basecase NLP to obtain the master NLP as
 * instructed by HiOp via @set_quadratic_regularization
 *  - (re)solve master NLP and return the primal optimal solution 'x' to HiOp via
 * @solve function.
 *
 * In addition, the user is required to implement 
 *     @eval_f_rterm 
 *     @eval_grad_rterm
 * to allow HiOp to compute function value and gradient vector individually for each recourse 
 * term  r_i. This will be done at arbitrary vectors 'x' that will be decided by HiOp. 
 */

class hiopInterfacePriDecProblem
{
public:
  /** 
   * Constructor
   */
  hiopInterfacePriDecProblem(MPI_Comm comm_world=MPI_COMM_WORLD)
  {
  }

  virtual ~hiopInterfacePriDecProblem()
  {
  }

  /** 
   * Solves the master problem consisting of the basecase problem plus the recourse terms.
   * The recourse terms have been added by the outer optimization loop (hiopAlgPrimalDecomposition)
   * via the 'add_' methods below
   * @param x : output, will contain the primal optimal solution of the master
   * 
   */
  virtual hiopSolveStatus solve_master(double* x,const bool& include_r, const double& rval=0, 
		                       const double* grad=0,const double*hess =0) = 0;

  virtual bool eval_f_rterm(size_t idx, const int& n, double* x, double& rval) = 0;
  virtual bool eval_grad_rterm(size_t idx, const int& n, double* x, double* grad) = 0;

  virtual bool set_quadratic_regularization(const int& n, const double* x, const double& rval,const double* grad,
		                    const double* hess) = 0;

  /** 
   * Returns the number S of recourse terms
   */
  virtual size_t get_num_rterms() const = 0;
  /**
   * Return the number of primal optimization variables
   */
  virtual size_t get_num_vars() const = 0;

  virtual void get_solution(double* x) const = 0;
  virtual double get_objective() = 0;
};
  
} //end of namespace
#endif
