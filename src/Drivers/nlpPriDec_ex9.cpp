#include "nlpPriDec_ex9.hpp"

#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"
/*
PriDecMasterProblemEx9::PriDecMasterProblemEx9(size_t nx, size_t ny, size_t nS, size_t S)
  : nx_(nx), ny_(ny), nS_(nS), S_(S)
{
  basecase_ = new PriDecBasecaseProblemEx9(nx_);

  //todo:
  // - generate S vectors \xi (each of size nS) from U[-0.25, 0.25]
  // - create S problems PriDecRecourseProblemEx9
}
PriDecMasterProblemEx9::~PriDecMasterProblemEx9()
{
  delete basecase_;
}
*/
using namespace hiop;
hiop::hiopSolveStatus
PriDecMasterProblemEx9::solve_master(double* x,
                                     const bool& include_r,
                                     const double& rval/*=0*/, 
                                     const double* grad/*=0*/,
                                     const double*hess /*=0*/)
{
  //todo:
  //  - solve PriDecBasecaseProblemEx9
  //  - retrieve whatever is needed from it

  obj_=-1e+20;
  hiopSolveStatus status;
  if(basecase_==NULL)
  {
    basecase_ = new PriDecBasecaseProblemEx9(nx_);
  }
  basecase_->set_include(include_r);

  if(include_r)
  {
    assert(basecase_->quad_is_defined());
  }
  
  hiopNlpSparse nlp(*basecase_);
    
  //nlp.options->SetStringValue("compute_mode", "hybrid");
  
  // what should the solver options be?
  nlp.options->SetStringValue("dualsUpdateType", "linear");
  nlp.options->SetStringValue("dualsInitialization", "zero");

  nlp.options->SetStringValue("fixed_var", "relax");
  nlp.options->SetStringValue("Hessian", "analytical_exact");
  nlp.options->SetStringValue("KKTLinsys", "xdycyd");
  //nlp.options->SetStringValue("compute_mode", "hybrid");
  //nlp.options->SetStringValue("mem_space", mem_space.c_str());

  nlp.options->SetIntegerValue("verbosity_level", 1);
  nlp.options->SetNumericValue("mu0", 1e-1);
  //nlp.options->SetNumericValue("tolerance", 1e-5);
  
  hiopAlgFilterIPMNewton solver(&nlp);

  status = solver.run();
  obj_ = solver.getObjective();
  solver.getSolution(x);

  if(status<0){
    printf("solver returned negative solve status: %d (with objective is %18.12e)\n", status, solver.getObjective());
    return status;
  }
   
  //for(int i=0;i<nx_;i++) printf(" %d %18.12e ",i,x[i]);
    
  if(sol_==NULL){
    sol_ = new double[nx_];
  }
  memcpy(sol_,x, nx_*sizeof(double));
  
  //compute the recourse estimate
  if(include_r)
  {
    double rec_appx = 0.;
    basecase_->get_rec_obj(nx_, x, rec_appx);
    printf("recourse estimate: is %18.12e\n", rec_appx);
  }

  return Solve_Success;
  //return hiop::SolverInternal_Error;
}

bool PriDecMasterProblemEx9::set_recourse_approx_evaluator(const int n, RecourseApproxEvaluator* evaluator)
{
  basecase_->set_quadratic_terms(n, evaluator);
  return true; 
}

bool PriDecMasterProblemEx9::eval_f_rterm(size_t idx, const int& n, const double* x, double& rval)
{
  // todo:
  //  - set `x` for the recourse PriDecRecourseProblemEx9 corresponding to `idx`
  //  - solve the problem
  //  - retrieve obj_value
  //  - retrieve/compute the gradient 
  assert(nx_==n);
  rval=-1e+20;
  hiopSolveStatus status;
  double* xi;
  xi = new double[ny_]; 
  for(int i=0;i<ny_;i++) xi[i] = 1.;
  xi[ny_-1] = 0.1;

  PriDecRecourseProblemEx9* ex9_recourse;

  //printf("ny %d\n",ny_); 
  //printf("nx %d\n",nx_); 

  ex9_recourse = new PriDecRecourseProblemEx9(ny_, nS_,S_,x,xi);
  //assert("for debugging" && false); //for debugging purpose
  hiopNlpMDS nlp(*ex9_recourse);

  nlp.options->SetStringValue("dualsUpdateType", "linear");
  nlp.options->SetStringValue("dualsInitialization", "zero");

  nlp.options->SetStringValue("Hessian", "analytical_exact");
  //nlp.options->SetStringValue("compute_mode", "hybrid");

  nlp.options->SetIntegerValue("verbosity_level", 0);
  nlp.options->SetNumericValue("mu0", 1e-1);
  //nlp.options->SetNumericValue("tolerance", 1e-5);

  hiopAlgFilterIPMNewton solver(&nlp);

  //assert("for debugging" && false); //for debugging purpose
  status = solver.run();
  rval = solver.getObjective();  
  if(y_==NULL){
    y_ = new double[ny_];
  }

  solver.getSolution(y_);
  return true;
      
};

bool PriDecMasterProblemEx9::eval_grad_rterm(size_t idx, const int& n, double* x, double* grad)
{
  //todo:
  // return in grad the gradient computed in eval_f_rterm
  assert(nx_==n);
  for(int i=0;i<n;i++) grad[i] = (x[i]-y_[i])/S_;
  return true;
};

