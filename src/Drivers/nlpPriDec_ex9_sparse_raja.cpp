#include "nlpPriDec_ex9_sparse_raja.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"

#include <RAJA/RAJA.hpp>

using ex9_raja_exec = hiop::hiop_raja_exec;
using ex9_raja_reduce = hiop::hiop_raja_reduce;
using namespace hiop;

PriDecMasterProblemEx9Sparse::
PriDecMasterProblemEx9Sparse(size_t nx, size_t ny, size_t nS, size_t S, std::string mem_space) 
  : nx_(nx), 
    ny_(ny),
    nS_(nS),
    S_(S),
    mem_space_(mem_space)
{
  assert(nx==ny);
  y_ = new double[ny_];
  sol_ = new double[nx_];
  obj_ = 1e20;
  basecase_ = new PriDecBasecaseProblemEx9(nx_);
  nc_ = nx_;
};

PriDecMasterProblemEx9Sparse::~PriDecMasterProblemEx9Sparse()
{
  delete[] y_;
  delete[] sol_; 
  delete basecase_;
};
	
hiop::hiopSolveStatus
PriDecMasterProblemEx9Sparse::solve_master(hiopVector& x,
                                           const bool& include_r,
                                           const double& rval/*=0*/, 
                                           const double* grad/*=0*/,
                                           const double*hess /*=0*/,
                                           const char* master_options_file/*=nullptr*/)
{
  obj_=-1e+20;
  hiopSolveStatus status;
  if(basecase_==nullptr) {
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
    nlp.log->printf(hovError, 
                    "solver returned negative solve status: %d (with objective is %18.12e)\n", 
                    status, solver.getObjective());
    return status;
  }
  
  if(sol_==nullptr) {
    sol_ = new double[nx_];
  }
  memcpy(sol_, x_vec, nx_*sizeof(double));
  
  //compute the recourse estimate
  if(include_r) {
    double rec_appx = 0.;
    basecase_->get_rec_obj(nx_, x_vec, rec_appx);
    //printf("recourse estimate: is %18.12e\n", rec_appx); 
  }
  
  return Solve_Success;
  //return hiop::SolverInternal_Error;
}

bool PriDecMasterProblemEx9Sparse::
set_recourse_approx_evaluator(const int n, 
		              hiopInterfacePriDecProblem::RecourseApproxEvaluator* evaluator)
{
  assert(n==nc_);
  basecase_->set_quadratic_terms(n, evaluator);
  return true; 
}

// all the memory management is done through umpire in the PriDecRecourseProblemEx9Sparse class
bool PriDecMasterProblemEx9Sparse::eval_f_rterm(size_t idx, const int& n, const double* x, double& rval)
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
  for(int i=0; i<nS_; i++) { 
    xi[i] = 1.; 
  }
  //xi[ny_-1] = double(idx/100.0+1.);

  PriDecRecourseProblemEx9Sparse* ex9_recourse;
  //printf("Constructor of recourse\n");

  ex9_recourse = new PriDecRecourseProblemEx9Sparse(nc_, nS_, S_, x, xi, mem_space_);
  
  //printf("Constructor of recourse complete\n");
  // set a few contingencies to have different sparse structure
  /*
  if(idx%30==0) {  
    ex9_recourse->set_sparse(0.3);
  }
  */
  
  hiopNlpSparse nlp(*ex9_recourse);
  nlp.options->SetStringValue("duals_update_type", "linear");
  //nlp.options->SetStringValue("dualsInitialization", "zero");
  nlp.options->SetStringValue("Hessian", "analytical_exact");
#ifdef HIOP_USE_GPU
  //nlp.options->SetStringValue("compute_mode", "hybrid");
  nlp.options->SetStringValue("compute_mode", "cpu");
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
  if(y_==nullptr) {
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
bool PriDecMasterProblemEx9Sparse::eval_grad_rterm(size_t idx, const int& n, double* x, hiopVector& grad)
{
  assert(nx_==n);
  double* grad_vec = grad.local_data();
  for(int i=0; i<n; i++) { 
    grad_vec[i] = (x[i]-y_[i]);
  }
  return true;
};

inline size_t PriDecMasterProblemEx9Sparse::get_num_rterms() const
{
  return S_;
}

inline size_t PriDecMasterProblemEx9Sparse::get_num_vars() const
{
  return nx_;
}

void PriDecMasterProblemEx9Sparse::get_solution(double* x) const
{
  assert(sol_!=nullptr);
  memcpy(x, sol_, nx_*sizeof(double));
};

double PriDecMasterProblemEx9Sparse::get_objective()
{
  return obj_;
};

