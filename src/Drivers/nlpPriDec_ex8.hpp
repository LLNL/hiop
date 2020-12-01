#ifndef HIOP_EXAMPLE_EX8
#define HIOP_EXAMPLE_EX8

#include "hiopInterfacePrimalDecomp.hpp"

#include "hiopInterface.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"
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
#include <chrono>

using namespace hiop;
/* 
class Ex8 : public hiop::hiopInterfaceMDS
{
public:
  Ex8(int ns_)
    : Ex8(ns_, ns_)  //ns = nx, nd=S
  {
  }
  
  Ex8(int ns_, int nd_)
    : ns(ns_)
  {
    if(ns<0) {
      ns = 0;
    } else {
      if(4*(ns/4) != ns) {
	ns = 4*((4+ns)/4);
	printf("[warning] number (%d) of sparse vars is not a multiple ->was altered to %d\n", 
	       ns_, ns); 
      }
    }

    if(nd_<0) nd=0;
    else nd = nd_;
    if(nd<ns){
      nd = ns;
      printf("[warning] number (%d) of recourse problems should be larger than sparse vars  %d,"
	     " changed to be the same\n",  nd, ns); 
    }
    //haveIneq = true;
  }

  virtual ~Ex8()
  {
  }
  
  bool get_prob_sizes(long long& n, long long& m)
  { 
    n=ns;
    m=0; 
    return true; 
  }

  bool get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type)
  {
    //assert(n>=4 && "number of variables should be greater than 4 for this example");
    assert(n==ns);
    //x
    for(int i=0; i<ns; ++i) xlow[i] = 0.;
    //x
    for(int i=0; i<ns; ++i) xupp[i] = +1e+20;

    for(int i=0; i<n; ++i) type[i]=hiopNonlinear;
    return true;
  }

  bool get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type)
  {
    assert(m==0);

    return true;
  }
  bool eval_f(const long long& n, const double* x, bool new_x, double& obj_value)
  {
    //assert(ns>=4);
    obj_value=0.;//x[0]*(x[0]-1.);
    //sum 0.5 {(x_i-1)*(x_{i}-1) : i=1,...,ns} + 0.5 y'*Qd*y + 0.5 s^T s
    for(int i=0; i<ns; i++) obj_value += (x[i]-1.)*(x[i]-1.);
    obj_value *= 0.5;

    if(include_r){
      for(int i=0; i<ns; i++) obj_value += 0.5*x[i]*x[i]+x[i]+0.5*ns;
    }

    return true;
  }
  virtual bool eval_cons(const long long& n, const long long& m, 
			 const long long& num_cons, const long long* idx_cons,  
			 const double* x, bool new_x, double* cons)
  {
    assert(num_cons==0);
    return true;
  }
 
  //sum 0.5 {(x_i-1)*(x_{i}-1) : i=1,...,ns} 
  bool eval_grad_f(const long long& n, const double* x, bool new_x, double* gradf)
  {
    //! assert(ns>=4); assert(Q->n()==ns/4); assert(Q->m()==ns/4);
    //x_i - 0.5 
    for(int i=0; i<ns; i++) 
      gradf[i] = x[i]-1.;
    if(include_r){
      for(int i=0; i<ns; i++) gradf[i] += x[i]+1.;
    }
    return true;
  }

  // Implementation of the primal starting point specification //
  bool get_starting_point(const long long& global_n, double* x0)
  {
    assert(global_n==ns); 
    for(int i=0; i<global_n; i++) x0[i]= 2.;
    return true;
  }
  bool get_starting_point(const long long& n, const long long& m,
				  double* x0,
				  bool& duals_avail,
				  double* z_bndL0, double* z_bndU0,
				  double* lambda0)
  {
    duals_avail = false;
    return false;
  }


  // pass the COMM_SELF communicator since this example is only intended to run inside 1 MPI process //
  virtual bool get_MPI_comm(MPI_Comm& comm_out) { comm_out=MPI_COMM_SELF; return true;}
  virtual bool
  eval_Jac_cons(const long long& n, const long long& m, 
		const long long& num_cons, const long long* idx_cons,
		const double* x, bool new_x,
		const long long& nsparse, const long long& ndense, 
		const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
		double** JacD)
  {
    assert(m==0);
    return true;
  }
  bool eval_Hess_Lagr(const long long& n, const long long& m, 
			      const double* x, bool new_x, const double& obj_factor,
			      const double* lambda, bool new_lambda,
			      const long long& nsparse, const long long& ndense, 
			      const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS, 
			      double** HDD,
			      int& nnzHSD, int* iHSD, int* jHSD, double* MHSD)
  {
    //Note: lambda is not used since all the constraints are linear and, therefore, do 
    //not contribute to the Hessian of the Lagrangian
    assert(nnzHSS==ns);
    assert(nnzHSD==0);
    assert(iHSD==NULL); assert(jHSD==NULL); assert(MHSD==NULL);

    if(iHSS!=NULL && jHSS!=NULL) {
      for(int i=0; i<ns; i++) iHSS[i] = jHSS[i] = i;     
    }

    if(MHSS!=NULL) {
      for(int i=0; i<ns; i++) MHSS[i] = obj_factor;
    }
    //assert(HDD==NULL);
    return true;
  }
  bool get_sparse_dense_blocks_info(int& nx_sparse, int& nx_dense,
				    int& nnz_sparse_Jace, int& nnz_sparse_Jaci,
				    int& nnz_sparse_Hess_Lagr_SS, int& nnz_sparse_Hess_Lagr_SD)
  {
    nx_sparse = ns;
    nx_dense = 0;
    nnz_sparse_Jace = 0;
    nnz_sparse_Jaci = 0;
    nnz_sparse_Hess_Lagr_SS = ns;
    nnz_sparse_Hess_Lagr_SD = 0.;
    return true;
  }


protected:
  int ns,nd;
  bool include_r = false;

};
*/

/* Problem test for the AlgPrimalDecomposition
 *  min   sum 0.5 {(x[i]-1)*(x[i]-1) : i=1,...,ns} + 0.5/S sum{i=1,...,S} R_i(x)
 *  where S>=ns
 *  for i=1,...,S
 *        R_i(x) = sum{j=1,..,ns}  0.5*(x[j]+S)(x[j]+S)   if j=i, j<=ns
 *                                 0.5*x[j]*x[j]
 *  s.t. x >= 0      
 *        
 * The example problem does not have constaints.       
 * S number of contingencies       
 *
 * Coding of the problem in DenseConstraints HiOp input: variable x
 * 
 */


class Ex8 : public hiop::hiopInterfaceDenseConstraints
{
public:
  Ex8(int ns_)
    : Ex8(ns_, ns_)  //ns = nx, nd=S
  {
  }
  
  Ex8(int ns_, int S_)
    : ns(ns_),evaluator_(NULL) 

  {
    if(ns<0) {
      ns = 0;
    } else {
      if(4*(ns/4) != ns) {
	ns = 4*((4+ns)/4);
	printf("[warning] number (%d) of sparse vars is not a multiple ->was altered to %d\n", 
	       ns_, ns); 
      }
    }

    if(S_<0) S=0;
    else S = S_;
    if(S<ns){
      S = ns;
      printf("[warning] number (%d) of recourse problems should be larger than sparse vars  %d,"
	     " changed to be the same\n",  S, ns); 
    }
    //haveIneq = true;
  }
  Ex8(int ns_, int S_, bool include_)
    : Ex8(ns_,S_)
  {
    include_r = include_;
    /*if(include_r){
      rval = 0.;
      rgrad = new double[ns];
      rhess = new double[ns];
      x0 = new double[ns];
    }*/
  }
  Ex8(int ns_, int S_, bool include, const RecourseApproxEvaluator* evaluator)
    : Ex8(ns_,S_)

  {
    include_r = include;
    evaluator_ = new RecourseApproxEvaluator(ns, S, evaluator->get_rval(), evaluator->get_rgrad(), 
		            evaluator->get_rhess(), evaluator->get_x0());
  }

  virtual ~Ex8()
  {
    delete[] evaluator_;
  }
  
  bool get_prob_sizes(long long& n, long long& m)
  { 
    n=ns;
    m=0; 
    return true; 
  }

  bool get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type)
  {
    //assert(n>=4 && "number of variables should be greater than 4 for this example");
    assert(n==ns);
    //x
    for(int i=0; i<ns; ++i) xlow[i] = 0.;
    //x
    for(int i=0; i<ns; ++i) xupp[i] = +1e+20;

    for(int i=0; i<ns; ++i) type[i]=hiopNonlinear;
    return true;
  }

  bool get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type)
  {
    assert(m==0);

    return true;
  }
  bool eval_f(const long long& n, const double* x, bool new_x, double& obj_value)
  {
    //assert(ns>=4);
    obj_value=0.;//x[0]*(x[0]-1.);
    //sum 0.5 {(x_i-1)*(x_{i}-1) : i=1,...,ns} + 0.5 y'*Qd*y + 0.5 s^T s
    for(int i=0; i<ns; i++) obj_value += (x[i]-1.)*(x[i]-1.);
    obj_value *= 0.5;

    if(include_r){
      assert(evaluator_->get_rgrad()!=NULL);
      evaluator_->eval_f(ns, x, new_x, obj_value);
    }

    return true;
  }
  virtual bool eval_cons(const long long& n, const long long& m, 
			 const long long& num_cons, const long long* idx_cons,  
			 const double* x, bool new_x, double* cons)
  {
    assert(num_cons==0);
    return true;
  }
 
  //sum 0.5 {(x_i-1)*(x_{i}-1) : i=1,...,ns} 
  bool eval_grad_f(const long long& n, const double* x, bool new_x, double* gradf)
  {
    //! assert(ns>=4); assert(Q->n()==ns/4); assert(Q->m()==ns/4);
    //x_i - 0.5 
    for(int i=0; i<ns; i++) 
      gradf[i] = x[i]-1.;
    if(include_r){
      assert(evaluator_->get_rgrad()!=NULL);
      evaluator_->eval_grad(ns, x, new_x, gradf);
    }
    return true;
  }

  // Implementation of the primal starting point specification //
  bool get_starting_point(const long long& global_n, double* x0_)
  {
    assert(global_n==ns); 
    for(int i=0; i<global_n; i++) x0_[i]= 2.;
    return true;
  }

  bool get_starting_point(const long long& n, const long long& m,
				  double* x0_,
				  bool& duals_avail,
				  double* z_bndL0, double* z_bndU0,
				  double* lambda0)
  {
    duals_avail = false;
    return false;
  }

  // pass the COMM_SELF communicator since this example is only intended to run inside 1 MPI process //
  virtual bool get_MPI_comm(MPI_Comm& comm_out) { comm_out=MPI_COMM_SELF; return true;}
  virtual bool
  eval_Jac_cons(const long long& n, const long long& m,
			const long long& num_cons, const long long* idx_cons,  
			const double* x, bool new_x, double** Jac) 
  {
    assert(m==0);
    return true;
  }

  bool quad_is_defined() {
    if(evaluator_!=NULL) return true;
    else return false;
  }

  bool set_quadratic_terms(const int& n, const RecourseApproxEvaluator* evaluator)
  {
    assert(ns == n);
    if(evaluator_==NULL){
    
      evaluator_ = new RecourseApproxEvaluator(n, S, evaluator->get_rval(), evaluator->get_rgrad(), 
		            evaluator->get_rhess(), evaluator->get_x0());
      return true;
    }

    if(evaluator->get_rgrad()!=NULL){
      evaluator_->set_rval(evaluator->get_rval());
     
      evaluator_->set_rgrad(ns,evaluator->get_rgrad());
      evaluator_->set_rhess(ns,evaluator->get_rhess());
      
      evaluator_->set_x0(ns,evaluator->get_x0());
    }
    return true;
  }
  bool set_include(bool include){
    include_r = include;
    return true;
  }

protected:
  int ns,S;
  bool include_r = false;
  RecourseApproxEvaluator* evaluator_;

};


class PriDecMasterProblemEx8 : public hiopInterfacePriDecProblem
{
public:
  PriDecMasterProblemEx8(int n,
                         int S,
                         MPI_Comm comm_world=MPI_COMM_WORLD)
    : hiopInterfacePriDecProblem(comm_world),
      n_(n), S_(S),obj_(-1e20),sol_(NULL)
  {
      my_nlp = new Ex8(n_,S_);   
  }

  virtual ~PriDecMasterProblemEx8()
  {
  }

  hiopSolveStatus solve_master(double* x, const bool& include_r, const double& rval = 0,
		               const double* grad=0, const double* hess=0)
  {

    //user's NLP -> implementation of hiop::hiopInterfaceMDS
    double obj_value=-1e+20;
    hiopSolveStatus status;
    //Ex8* my_nlp;
    //my_nlp = new Ex8(12);
    //hiopNlpMDS nlp(*my_nlp);
    //Ex8 nlp_interface(n);
    //hiopNlpDenseConstraints nlp(*my_nlp);
    if(my_nlp==NULL){
      my_nlp = new Ex8(n_,S_);
    }

    printf("here2\n");

    bool ierr = my_nlp->set_include(include_r);
    if(include_r){
      assert(my_nlp->quad_is_defined());
    }
      // check to see if the resource value and gradient are correct
      //printf("recourse value: is %18.12e)\n", rval_);
      //for(int i=0;i<n_;i++) printf("%d %18.12e\n",i,rgrad_[i]);
      //assert("for debugging" && false); //for debugging purpose
    //if(rank==0) printf("interface created\n");
    hiopNlpDenseConstraints nlp(*my_nlp);
    //if(rank==0) printf("nlp formulation created\n");  

    /*
    nlp.options->SetStringValue("dualsUpdateType", "linear");
    nlp.options->SetStringValue("dualsInitialization", "zero");

    nlp.options->SetStringValue("Hessian", "analytical_exact");
    nlp.options->SetStringValue("KKTLinsys", "xdycyd");
    nlp.options->SetStringValue("compute_mode", "hybrid");

    nlp.options->SetIntegerValue("verbosity_level", 3);
    nlp.options->SetNumericValue("mu0", 1e-1);
    nlp.options->SetNumericValue("tolerance", 1e-5);
    */
    hiopAlgFilterIPM solver(&nlp);
    status = solver.run();
    obj_value = solver.getObjective();
    solver.getSolution(x);

    if(status<0) {
      printf("solver returned negative solve status: %d (with objective is %18.12e)\n", status, obj_value);
      return status;
    }
    for(int i=0;i<n_;i++) printf("%d %18.12e\n",i,x[i]);
    //pretend that the master problem has all zero solution
    /*
    if(include_r==0){
      for(int i=0; i<n_; i++)
        x[i] = 1.;
    }
    else{
      for(int i=0; i<n_; i++)
        x[i] = 0.; 
    }
    */
    if(sol_==NULL){
      sol_ = new double[n_];
    }
    memcpy(sol_,x, n_*sizeof(double));

    return Solve_Success;
  };
  // The recourse solution is 0.5*(x+Se_i)(x+Se_i)
  bool eval_f_rterm(size_t idx, const int& n, double* x, double& rval)
  {
    rval = 0.;
    for(int i=0; i<n; i++) {
      if(i==idx)	    
	rval += (x[i]+S_)*(x[i]+S_);
      else
	rval += x[i]*x[i];
    }
    rval *= 0.5;
    rval /= S_;
    return true;
  }
  bool eval_grad_rterm(size_t idx, const int& n, double* x, double* grad)
  {
    assert(n_ == n);
    for(int i=0; i<n; i++){
      if(i==idx)	    
        grad[i] = (x[i]+S_)/S_;
      else
	grad[i] = x[i]/S_;
    }
    return true;
  } 


  //implement with alpha = 1 for now only
  // this function should only be used if quadratic regularization is included
  bool set_recourse_approx_evaluator(const int n, RecourseApproxEvaluator* evaluator){
  
     my_nlp->set_quadratic_terms( n, evaluator);
     return true; 
  }


  /*
  bool set_quadratic_regularization(RecourseApproxEvaluator* evaluator)
		  //const int& n, const double* x0, const double& rval,const double* grad,
		  //                  const double* hess)
  {
    assert(n_ == n);
    my_nlp->set_quadratic_terms(n,true,evaluator);
    rval_ = rval;
    if(rgrad_==NULL){
      rgrad_ = new double[n_];
    }
    if(rhess_==NULL){
      rhess_ = new double[n_];
    }
    if(x0_==NULL){
      x0_ = new double[n_];
    }
    memcpy(rgrad_,grad , n_*sizeof(double));
    memcpy(rhess_,hess , n_*sizeof(double));
    memcpy(x0_,x0, n_*sizeof(double));
    
    return true;
  }
  */
  /** 
   * Returns the number S of recourse terms
   */
  size_t get_num_rterms() const
  {
    return S_;
  }
  size_t get_num_vars() const
  {
    return n_;
  }
  void get_solution(double* x) const
  {
    for(int i=0; i<n_; i++)
      x[i] = sol_[i];
  }

  double get_objective()
  {
    return 0.;
  }
private:
  size_t n_;
  size_t S_;
  Ex8* my_nlp;
  //double rval_;
  //double* rgrad_;
  //double* rhess_; 
  //double* x0_;
  double obj_;
  double* sol_; 
  // will need some encapsulation of the basecase NLP
  // nlpMDSForm_ex4.hpp
};

#endif
