#ifndef HIOP_INTERFACE_PRIDEC
#define HIOP_INTERFACE_PRIDEC

#include "hiopInterface.hpp"
#include "hiopVector.hpp"
#include "hiopLinAlgFactory.hpp"
#include <cassert>
#include <cstring> //for memcpy
#include <vector>

namespace hiop
{

/** 
 * Base class (interface) for specifying optimization NLPs that have separable terms in the 
 * objective, which we coin as "primal decomposable" problems. More specifically, these problems 
 * have the following structure (please also take a note of the terminology):
 *
 *   min_x  basecase(x) + 1/S sum { r_i(x) : i=1,...,S}      (primal decomposable NLP)
 *
 * The subproblem <b>'basecase'</b> refers to a general nonlinear nonconvex NLP in `x`. We point out 
 * that the basecase can have general <i>twice continously differentiable</i> objective and 
 * constraints; the latter can be equalities, inequalities, and bounds on `x`.
 * Furthermore, borrowing from stochastic programming terminology, the terms `r_i` are 
 * called <em> recourse terms </em>, or, in short, <b> r-terms </b>.
 * 
 * In order to solve the above problem, HiOp solver will perform a series of approximations and will
 * require the user to solve a so-called 'master' problem
 * 
 *   min  basecase(x) +  q(x)                            (master NLP)
 *    x
 * where the function q(x) is a convex differentiable approximation of sum { r_i(x) : i=1,...,S}
 * that we refer to as <i> quadratic regularization </i>.
 * 
 * The user is required to maintain and solve the master problem, more specifically:
 *  - to add the quadratic regularization to the basecase NLP; the quadratic regularization is 
 * provided by HiOp hiopInterfacePriDecProblem::RecourseApproxEvaluator classs. the user is 
 * expected to implement hiopInterfacePriDecProblem::set_recourse_approx_evaluator in the master 
 * problem class.
 *  - to (re)solve master NLP  and return the primal optimal solution `x` to HiOp; for doing this, 
 * the user is required to implement hiopInterfacePriDecProblem::solve_master method.
 *
 * In addition, the user is required to implement 
 *     - hiopInterfacePriDecProblem::eval_f_rterm 
 *     - hiopInterfacePriDecProblem::eval_grad_rterm
 * which solves the individual recourse subproblems.
 *
 * These methods will be used by the HiOp's primal decomposition solver to  compute function value 
 * and  gradient vector individually for each recourse term  r_i, which are needed to build the 
 * convex regularizations q(x). The above methods will be called at arbitrary vectors `x` that 
 * are decided internally  by HiOp. 
 *
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
  virtual hiopSolveStatus solve_master(hiopVector& x,const bool& include_r, const double& rval=0, 
		                       const double* grad=0,const double*hess =0) = 0;

  virtual bool eval_f_rterm(size_t idx, const int& n, const double* x, double& rval) = 0;
  virtual bool eval_grad_rterm(size_t idx, const int& n, double* x, hiopVector& grad) = 0;

  //
  // Documentation here
  //
  //virtual bool set_quadratic_regularization(const int& n, const double* x, const double& rval,const double* grad,
  //		                    const double* hess) = 0;
  
  //virtual bool set_recourse_approx_evaluator(const int n, int S, const std::vector<int>& list,
  //                                             const double& rval, const double* rgrad, 
  //		                               const double* rhess, const double* x0);


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


  /** 
   * Define the evaluator class called by the base case problem class to add the quadratic 
   * recourse approximation
   * This class is intened for internal use of hiopInterfacePriDecProblem class only
   * In the cases where only RecourseApproxEvaluator is needed, a shell hiopInterfacePriDecProblem
   * is still required to be created.
   */
  class RecourseApproxEvaluator
  {
  public:
    RecourseApproxEvaluator(int nc);
 
    RecourseApproxEvaluator(int nc, int S);
    
    RecourseApproxEvaluator(const int nc, const int S, const int* list);
  
    RecourseApproxEvaluator(const int nc, const int S, const double& rval, const hiopVector& rgrad, 
                            const hiopVector& rhess, const hiopVector& x0);
  
    RecourseApproxEvaluator(int nc,int S, const int* list,
                            const double& rval, const hiopVector& rgrad, 
                            const hiopVector& rhess, const hiopVector& x0);

    ~RecourseApproxEvaluator();

    bool eval_f(const long long& n, const double* x, bool new_x, double& obj_value);
 
    bool eval_grad(const long long& n, const double* x, bool new_x, double* grad);

    bool eval_hess(const long long& n, const hiopVector& x, bool new_x, hiopVector& hess);
  
    virtual bool get_MPI_comm(MPI_Comm& comm_out);
    
    void set_rval(const double rval);
    void set_rgrad(const int n, const hiopVector& rgrad);
    void set_rhess(const int n, const hiopVector& rhess);
    void set_x0(const int n, const hiopVector& x0);
    void set_xc_idx(const int* idx);

    int get_S() const; 
    double get_rval() const;
    hiopVector* get_rgrad() const;
    hiopVector* get_rhess() const;
    hiopVector* get_x0() const; 
    int* get_xc_idx() const; 

  protected:
    int nc_,S_;
    int* xc_idx_;
    double rval_;
    //double* rgrad_;
    hiopVector* rgrad_;
    hiopVector* rhess_; //diagonal Hessian vector
    hiopVector* x0_; //current solution
  };
  
  virtual bool set_recourse_approx_evaluator(const int n, RecourseApproxEvaluator* evaluator)=0;
  
};
  
} //end of namespace
#endif
