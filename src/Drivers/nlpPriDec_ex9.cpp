#include "nlpPriDec_ex9.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"

/*
  todo:
  - generate S vectors \xi (each of size nS) from U[-0.25, 0.25]
*/
using namespace hiop;

	
PriDecMasterProblemEx9::
PriDecMasterProblemEx9(size_t nx, size_t ny, size_t nS, size_t S) : nx_(nx), ny_(ny),nS_(nS),S_(S)
{
  assert(nx==ny);
  y_ = new double[ny_];
  sol_ = new double[nx_];
  obj_ = 1e20;
  basecase_ = new PriDecBasecaseProblemEx9(nx_);
  nc_ = nx_;
};

PriDecMasterProblemEx9::~PriDecMasterProblemEx9()
{
  delete[] y_;
  delete[] sol_; 
  delete basecase_;
};
	
hiop::hiopSolveStatus
PriDecMasterProblemEx9::solve_master(hiopVector& x,
                                     const bool& include_r,
                                     const double& rval/*=0*/, 
                                     const double* grad/*=0*/,
                                     const double*hess /*=0*/,
                                     const char* master_options_file/*=nullptr*/)
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
  
  hiopNlpSparse nlp(*basecase_, master_options_file);

  //
  // any of the options below can be overwritten by specifying them in the 'hiop_pridec_master.options' file
  //
  //nlp.options->SetStringValue("compute_mode", "hybrid");
  //nlp.options->SetStringValue("dualsUpdateType", "linear");

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
  double* x_vec = x.local_data();
  solver.getSolution(x_vec);

  if(status<0) {
    printf("solver returned negative solve status: %d (with objective is %18.12e)\n", status, solver.getObjective());
    return status;
  }
  
  // print x
  // for(int i=0;i<nx_;i++) printf(" %d %18.12e ",i,x[i]);
    
  if(sol_==NULL) {
    sol_ = new double[nx_];
  }
  memcpy(sol_,x_vec, nx_*sizeof(double));
  
  //compute the recourse estimate
  if(include_r) {
    double rec_appx = 0.;
    basecase_->get_rec_obj(nx_, x_vec, rec_appx);
    //printf("recourse estimate: is %18.12e\n", rec_appx); 
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
  
#ifdef HIOP_USE_MPI
  double t3 =  MPI_Wtime(); 
  double t4 = 0.; 
#endif 
  
  // xi can be set below 
  xi = new double[nS_]; 
  for(int i=0;i<nS_;i++) xi[i] = 1.;
  //xi[ny_-1] = double(idx/100.0+1.);

  PriDecRecourseProblemEx9* ex9_recourse;

  ex9_recourse = new PriDecRecourseProblemEx9(nc_, nS_,S_,x,xi);
  
  // set a few contingencies to have different sparse structure
  /*
  if(idx%30==0) {  
    ex9_recourse->set_sparse(0.3);
  }
  */
  
  hiopNlpMDS nlp(*ex9_recourse);
  nlp.options->SetStringValue("duals_update_type", "linear");
  //nlp.options->SetStringValue("dualsInitialization", "zero");
  nlp.options->SetStringValue("Hessian", "analytical_exact");
#ifdef HIOP_USE_GPU
  nlp.options->SetStringValue("compute_mode", "hybrid");
#else  
  nlp.options->SetStringValue("compute_mode", "cpu");
#endif
  //nlp.options->SetStringValue("time_kkt", "on");
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
  // uncomment if want to monitor contingency computing time
  /* t4 =  MPI_Wtime(); 
     if(idx==0||idx==1) {
       printf( "Elapsed time for contingency %d is %f\n",idx, t4 - t3 ); 
       printf(" Objective for idx %d value %18.12e, xi %18.12e\n",idx,rval,xi[0]);
     }
  */
  #endif

  delete[] xi;
  delete ex9_recourse;
  return true;
};

//returns the gradient computed in eval_f_rterm
bool PriDecMasterProblemEx9::eval_grad_rterm(size_t idx, const int& n, double* x, hiopVector& grad)
{
  assert(nx_==n);
  double* grad_vec = grad.local_data();
  for(int i=0;i<n;i++) { 
    grad_vec[i] = (x[i]-y_[i]);
  }
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

