#include "nlpPriDec_ex8.hpp"

#include <cassert>
#include <cstring> //for memcpy
#include <cstdio>
#include <cmath>


bool Ex8::get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type)
{
  //assert(n>=4 && "number of variables should be greater than 4 for this example");
  assert(n==ns);
  //x
  for(int i=0; i<ns; ++i) xlow[i] = 0.;
  for(int i=0; i<ns; ++i) xupp[i] = +1e+20;
  for(int i=0; i<ns; ++i) type[i]=hiopNonlinear;
  return true;
};


bool Ex8::get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type)
{
  assert(m==0);
  return true;
};

bool Ex8::eval_f(const long long& n, const double* x, bool new_x, double& obj_value)
{
  obj_value=0.;//x[0]*(x[0]-1.);
  //sum 0.5 {(x_i-1)*(x_{i}-1) : i=1,...,ns} + 0.5 y'*Qd*y + 0.5 s^T s
  for(int i=0; i<ns; i++) obj_value += (x[i]-1.)*(x[i]-1.);
  obj_value *= 0.5;

  if(include_r)
  {
    assert(evaluator_->get_rgrad()!=NULL);
    evaluator_->eval_f(ns, x, new_x, obj_value);
  }
  return true;
};


bool Ex8::eval_cons(const long long& n, const long long& m, 
                       const long long& num_cons, const long long* idx_cons,  
		       const double* x, bool new_x, double* cons)
{
  assert(num_cons==0);
  return true;
};
 
bool Ex8::eval_grad_f(const long long& n, const double* x, bool new_x, double* gradf)
{
  //! assert(ns>=4); assert(Q->n()==ns/4); assert(Q->m()==ns/4);
  //x_i - 0.5 
  for(int i=0; i<ns; i++)
  {
    gradf[i] = x[i]-1.;
  }
  if(include_r)
  {
    assert(evaluator_->get_rgrad()!=NULL);
    evaluator_->eval_grad(ns, x, new_x, gradf);
  }
  return true;
};

bool Ex8::get_starting_point(const long long& global_n, double* x0_)
{
  assert(global_n==ns); 
  for(int i=0; i<global_n; i++) x0_[i]= 2.;
  return true;
};

bool Ex8::get_starting_point(const long long& n, const long long& m,
				  double* x0_,
				  bool& duals_avail,
				  double* z_bndL0, double* z_bndU0,
				  double* lambda0)
{
  duals_avail = false;
  return false;
};


bool Ex8::get_MPI_comm(MPI_Comm& comm_out) { comm_out=MPI_COMM_SELF; return true;};

bool Ex8::eval_Jac_cons(const long long& n, const long long& m,
	                        const long long& num_cons, const long long* idx_cons,  
			        const double* x, bool new_x, double* Jac) 
{
  assert(m==0);
  return true;
};

bool Ex8::quad_is_defined()
{
  if(evaluator_!=NULL) return true;
  else return false;
};

bool Ex8::set_quadratic_terms(const int& n, const RecourseApproxEvaluator* evaluator)
{
  assert(ns == n);
  if(evaluator_==NULL)
  {
    evaluator_ = new RecourseApproxEvaluator(n, S, evaluator->get_rval(), evaluator->get_rgrad(), 
                                             evaluator->get_rhess(), evaluator->get_x0());
    return true;
  }

  if(evaluator->get_rgrad()!=NULL)
  {
    evaluator_->set_rval(evaluator->get_rval());    
    evaluator_->set_rgrad(ns,evaluator->get_rgrad());
    evaluator_->set_rhess(ns,evaluator->get_rhess());  
    evaluator_->set_x0(ns,evaluator->get_x0());
  }
  return true;
};

bool Ex8::set_include(bool include)
{
  include_r = include;
  return true;
};

hiopSolveStatus PriDecMasterProblemEx8::solve_master(double* x, const bool& include_r, const double& rval/* = 0*/,
		                                     const double* grad/*=0*/, const double* hess/*=0*/)
{
  obj_=-1e+20;
  hiopSolveStatus status;
  if(my_nlp==NULL)
  {
    my_nlp = new Ex8(n_,S_);
  }

  bool ierr = my_nlp->set_include(include_r);
  if(include_r){
    assert(my_nlp->quad_is_defined());
  }
  // check to see if the resource value and gradient are correct
  //printf("recourse value: is %18.12e)\n", rval_);
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


  nlp.options->SetNumericValue("mu0", 1e-1);
  nlp.options->SetNumericValue("tolerance", 1e-5);
  */
  nlp.options->SetIntegerValue("verbosity_level", 1);
  hiopAlgFilterIPM solver(&nlp);
  status = solver.run();
  obj_ = solver.getObjective();
  solver.getSolution(x);

  if(status<0){
    printf("solver returned negative solve status: %d (with objective is %18.12e)\n", status, obj_);
    return status;
  }
  //for(int i=0;i<n_;i++) printf("%d %18.12e\n",i,x[i]);
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

bool PriDecMasterProblemEx8::eval_f_rterm(size_t idx, const int& n,const  double* x, double& rval)
{
  rval = 0.;
  for(int i=0; i<n; i++) {
    if(i==idx){	    
      rval += (x[i]+S_)*(x[i]+S_);
    }else{
      rval += x[i]*x[i];
    }
  }
  rval *= 0.5;
  rval /= S_;
  return true;
};

bool PriDecMasterProblemEx8::eval_grad_rterm(size_t idx, const int& n, double* x, double* grad)
{
  assert(n_ == n);
  for(int i=0; i<n; i++){
    if(i==idx){	    
      grad[i] = (x[i]+S_)/S_;
    }else{
      grad[i] = x[i]/S_;
    }
  }
  return true;
}; 

bool PriDecMasterProblemEx8::set_recourse_approx_evaluator(const int n, RecourseApproxEvaluator* evaluator)
{  
   my_nlp->set_quadratic_terms( n, evaluator);
   return true; 
}







