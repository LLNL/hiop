#ifndef HIOP_INTERFACE_PRIDEC
#define HIOP_INTERFACE_PRIDEC

//for solve status
#include "hiopInterface.hpp"
#include <cassert>
#include <cstring> //for memcpy

#ifdef HIOP_USE_MPI
#include "mpi.h"

#else
#ifndef MPI_Comm
#define MPI_Comm int
#endif

#ifndef MPI_COMM_WORLD
#define MPI_COMM_WORLD 0
#endif 

#ifndef MPI_COMM_SELF 
#define MPI_COMM_SELF 0
#endif 
#endif



namespace hiop
{
// todo: it looks like this class should call the "user" methods
//       eval_f_rterm  eval_grad_rterm
//       (and then build the approximation)
//   

// todo: move RecourseApproxEvaluator "inside" hiopInterfacePriDecProblem

class RecourseApproxEvaluator
{
public:

  RecourseApproxEvaluator(int ns_)
    : RecourseApproxEvaluator(ns_, ns_)  //ns = nx, nd=S
  {
  }
  RecourseApproxEvaluator(int ns, int S)
    : ns_(ns),S_(S),rval_(0.), rgrad_(NULL), rhess_(NULL),x0_(NULL)//ns = nx, nd=S
  {
     assert(S>=ns);
  }
  RecourseApproxEvaluator(int ns,int S, const double& rval, const double* rgrad, 
		         const double* rhess, const double* x0): ns_(ns),S_(S)
  {
    //assert(S>=ns);
    rval_ = rval;
    rgrad_ = new double[ns];
    rhess_ = new double[ns];
    x0_ = new double[ns];
    memcpy(rgrad_,rgrad, ns*sizeof(double));
    memcpy(rhess_,rhess , ns*sizeof(double));
    memcpy(x0_,x0, ns*sizeof(double));
  }


  bool eval_f(const long long& n, const double* x, bool new_x, double& obj_value)
  {
    //assert(ns>=4);
    //obj_value=0.;//x[0]*(x[0]-1.);
    //sum 0.5 {(x_i-1)*(x_{i}-1) : i=1,...,ns} + 0.5 y'*Qd*y + 0.5 s^T s
    assert(rgrad_!=NULL);
    obj_value += rval_;
    for(int i=0; i<ns_; i++) obj_value += rgrad_[i]*(x[i]-x0_[i]);
    for(int i=0; i<ns_; i++) obj_value += 0.5*rhess_[i]*(x[i]-x0_[i])*(x[i]-x0_[i]);

    return true;
  }
 
  //sum 0.5 {(x_i-1)*(x_{i}-1) : i=1,...,ns} 
  bool eval_grad(const long long& n, const double* x, bool new_x, double* grad)
  {
    //x_i - 0.5 
    assert(rgrad_!=NULL);
    for(int i=0; i<ns_; i++) grad[i] += rgrad_[i]+rhess_[i]*(x[i]-x0_[i]);
    return true;
  }

  bool eval_hess(const long long& n, const double* x, bool new_x, double* hess)
  {
    //x_i - 0.5 
    assert(rgrad_!=NULL);
    for(int i=0; i<ns_; i++) hess[i] += rhess_[i];
    return true;
  }
  // pass the COMM_SELF communicator since this example is only intended to run inside 1 MPI process //
  virtual bool get_MPI_comm(MPI_Comm& comm_out) { comm_out=MPI_COMM_SELF; return true;}
  
  void set_rval(const double rval){rval_ = rval;}
  void set_rgrad(const int n,const double* rgrad){
    assert(n == ns_);
    if(rgrad_==NULL){
      rgrad_ = new double[ns_];
    }
    memcpy(rgrad_,rgrad , ns_*sizeof(double));
  }
  void set_rhess(const int n,const double* rhess){
    assert(n == ns_);
    if(rgrad_==NULL){
      rhess_ = new double[ns_];
    }
    memcpy(rhess_,rhess , ns_*sizeof(double));
  }
  void set_x0(const int n,const double* x0){
    assert(n == ns_);
    if(rgrad_==NULL){
      rhess_ = new double[ns_];
    }
    memcpy(x0_,x0 , ns_*sizeof(double));
  }
  
  int get_S() const {return S_;} 
  double get_rval() const {return rval_;}
  double* get_rgrad() const {return rgrad_;}
  double* get_rhess() const {return rhess_;}
  double* get_x0() const {return x0_;}
  /*
  bool set_recourse_approx_quadratic(const int& n, const double* x0=NULL, 
		           const double& rval=0.,const double* grad=NULL,const double* hess=NULL)
  {
    assert(ns_ == n);
    if(grad!=NULL){
      rval_ = rval;
      if(rgrad_==NULL){
        rgrad_ = new double[ns_];
      }
      if(rhess_==NULL){
        rhess_ = new double[ns_];
      }
      if(x0_==NULL){
        x0_ = new double[ns_];
      }
      memcpy(rgrad_,grad , ns_*sizeof(double));
      memcpy(rhess_,hess , ns_*sizeof(double));
      memcpy(x0_,x0, ns_*sizeof(double));
    }
    return true;
  }
  */
protected:
  int ns_,S_;
  double rval_;
  double* rgrad_;
  double* rhess_; //diagonal vector for now
  double* x0_; //current solution
  //friend class Ex8;
};


/**
=======
namespace hiop
{
/** 
>>>>>>> 705b3825e670e9325445af09fa8df588b3e6ab32
 * Base class (interface) for specifying optimization NLPs that have separable terms in the 
 * objective, which we coin as "primal decomposable" problems. More specifically, these problems 
 * have the following structure (please also take a note of the terminology):
 *
 *   min_x  basecase(x) + sum { r_i(x) : i=1,...,S}      (primal decomposable NLP)
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
 * provided by HiOp hiopInterfacePriDecProblem::set_recourse_approx_evaluator
 *  - to (re)solve master NLP  and return the primal optimal solution `x` to HiOp; for doing this, 
 * the user is required to implement hiopInterfacePriDecProblem::solve_master method.
 *
 * In addition, the user is required to implement 
 *     - hiopInterfacePriDecProblem::eval_f_rterm 
 *     - hiopInterfacePriDecProblem::eval_grad_rterm
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
  virtual hiopSolveStatus solve_master(double* x,const bool& include_r, const double& rval=0, 
		                       const double* grad=0,const double*hess =0) = 0;

  virtual bool eval_f_rterm(size_t idx, const int& n, const double* x, double& rval) = 0;
  virtual bool eval_grad_rterm(size_t idx, const int& n, double* x, double* grad) = 0;

  //
  // Documentation here
  //
  //virtual bool set_quadratic_regularization(const int& n, const double* x, const double& rval,const double* grad,
  //		                    const double* hess) = 0;
  virtual bool set_recourse_approx_evaluator(const int n, RecourseApproxEvaluator* evaluator)=0;

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
