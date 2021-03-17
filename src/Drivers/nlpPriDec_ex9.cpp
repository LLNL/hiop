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


  obj_=-1e+20;
  hiopSolveStatus status;
  if(basecase_==NULL) {
    basecase_ = new PriDecBasecaseProblemEx9(nx_);
  }
  basecase_->set_include(include_r);

  if(include_r) {
    assert(basecase_->quad_is_defined());
  }
  
  hiopNlpSparse nlp(*basecase_);

  //if(include_r){
  //  assert("for debugging" && false); //for debugging purpose
 // }
    
  //nlp.options->SetStringValue("compute_mode", "hybrid");
  // what should the solver options be?
  //nlp.options->SetStringValue("dualsUpdateType", "linear");
  //nlp.options->SetStringValue("dualsInitialization", "zero"); hiop does not understand?

  //nlp.options->SetStringValue("fixed_var", "relax");
  nlp.options->SetStringValue("Hessian", "analytical_exact");
  nlp.options->SetStringValue("KKTLinsys", "xdycyd");
  nlp.options->SetStringValue("compute_mode", "cpu");
  //nlp.options->SetStringValue("compute_mode", "hybrid");
  //nlp.options->SetStringValue("mem_space", mem_space.c_str());

  nlp.options->SetIntegerValue("verbosity_level", 1);
  nlp.options->SetNumericValue("mu0", 1e-1);
  //nlp.options->SetNumericValue("tolerance", 1e-5);
  
  hiopAlgFilterIPMNewton solver(&nlp);


  status = solver.run();


  obj_ = solver.getObjective();
  solver.getSolution(x);

  if(status<0) {
    printf("solver returned negative solve status: %d (with objective is %18.12e)\n", status, solver.getObjective());
    return status;
  }
   
  //for(int i=0;i<nx_;i++) printf(" %d %18.12e ",i,x[i]);
    
  if(sol_==NULL) {
    sol_ = new double[nx_];
  }
  memcpy(sol_,x, nx_*sizeof(double));
  

  
  //compute the recourse estimate
  if(include_r) {
    double rec_appx = 0.;
    basecase_->get_rec_obj(nx_, x, rec_appx);
    printf("recourse estimate: is %18.12e\n", rec_appx);
  }
  

  return Solve_Success;
  //return hiop::SolverInternal_Error;
}

bool PriDecMasterProblemEx9::
set_recourse_approx_evaluator(const int n, 
		              hiopInterfacePriDecProblem::RecourseApproxEvaluator* evaluator)
{
  assert(n==nc_);
  basecase_->set_quadratic_terms(n, evaluator);
  return true; 
}

bool PriDecMasterProblemEx9::eval_f_rterm(size_t idx, const int& n, const double* x, double& rval)
{
  assert(nx_==n);
  rval=-1e+20;
  hiopSolveStatus status;
  double* xi;
  int nS_2 = nS_;

  #ifdef HIOP_USE_MPI
    double t3 =  MPI_Wtime(); 
    double t4 = 0.; 
  #endif 
  //if(idx==0||idx==50||idx==75||idx==150) {  
  //  nS_2 = 1;
  //}
  //if(idx<ny_ && idx>1) {
  //  nS_2 = idx;
  //}
  xi = new double[nS_2]; 
  //for(int i=0;i<nS_2;i++) xi[i] = idx*1./nS_2*5.-5.;
  for(int i=0;i<nS_2;i++) xi[i] = 1.;

  //xi[ny_-1] = double(idx/100.0+1.);

  PriDecRecourseProblemEx9* ex9_recourse;

  ex9_recourse = new PriDecRecourseProblemEx9(nc_, nS_2,S_,x,xi);
  
  if(idx%30==0) {  
    ex9_recourse->set_sparse(0.3);
  }
  
  //assert("for debugging" && false); //for debugging purpose
  hiopNlpMDS nlp(*ex9_recourse);

  nlp.options->SetStringValue("duals_update_type", "linear");
  //nlp.options->SetStringValue("dualsInitialization", "zero");

  nlp.options->SetStringValue("Hessian", "analytical_exact");
  nlp.options->SetStringValue("compute_mode", "cpu");
  //nlp.options->SetStringValue("compute_mode", "hybrid");
  nlp.options->SetStringValue("time_kkt", "on");
  nlp.options->SetIntegerValue("verbosity_level", 1);
  nlp.options->SetNumericValue("mu0", 1e-1);
  //nlp.options->SetNumericValue("tolerance", 1e-5);

  hiopAlgFilterIPMNewton solver(&nlp);

  //assert("for debugging" && false); //for debugging purpose
  status = solver.run();
  rval = solver.getObjective();  
  if(y_==NULL) {
    y_ = new double[ny_];
  }

  solver.getSolution(y_);
  
  #ifdef HIOP_USE_MPI
    t4 =  MPI_Wtime(); 
    if(idx==0||idx==1) {
      printf( "Elapsed time for contingency %d is %f\n",idx, t4 - t3 ); 
      printf(" Objective for idx %d value %18.12e, xi %18.12e\n",idx,rval,xi[0]);
    }
  #endif

  delete[] xi;
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

inline size_t PriDecMasterProblemEx9::get_num_rterms() const
{
  return S_;
}

inline size_t PriDecMasterProblemEx9::get_num_vars() const
{
  return nx_;
}

void PriDecMasterProblemEx9::get_solution(double* x) const
{
  assert(sol_!=NULL);
  memcpy(x,sol_, nx_*sizeof(double));
};

double PriDecMasterProblemEx9::get_objective()
{
  return obj_;
};

