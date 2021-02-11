#include "nlpMDSForm_raja_ex4.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"

#ifdef HIOP_USE_MAGMA
#include "magma_v2.h"
#endif

#include <cstdlib>
#include <string>

#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>
#include <RAJA/RAJA.hpp>


using namespace hiop;

static bool parse_arguments(int argc, char **argv,
			    bool& self_check,
			    long long& n_sp,
			    long long& n_de,
			    bool& one_call_cons)
{
  self_check=false;
  n_sp = 1000;
  n_de = 1000;
  one_call_cons = false;
  switch(argc) {
  case 1:
    //no arguments
    return true;
    break;
  case 5: // 4 arguments
    {
      if(std::string(argv[4]) == "-selfcheck")
	self_check=true;
    }
  case 4: // 3 arguments
    {
      one_call_cons = (bool) atoi(argv[3]);
    }
  case 3: //2 arguments
    {
      n_de = atoi(argv[2]);
      if(n_de<0) n_de = 0;
    }
  case 2: //1 argument
    {
      n_sp = atoi(argv[1]);
      if(n_sp<0) n_sp = 0;
    }
    break;
  default: 
    return false; //5 or more arguments
  }

  if(self_check && n_sp!=400 && n_de!=100)
    return false;
  
  return true;
};

static void usage(const char* exeName)
{
  printf("HiOp driver %s that solves a synthetic problem of variable size in the "
	 "mixed dense-sparse formulation.\n", exeName);
  printf("Usage: \n");
  printf("  '$ %s sp_vars_size de_vars_size eq_ineq_combined_nlp -selfcheck'\n", exeName);
  printf("Arguments, all integers, excepting string '-selfcheck'\n");
  printf("  'sp_vars_size': # of sparse variables [default 400, optional]\n");
  printf("  'de_vars_size': # of dense variables [default 100, optional]\n");
  printf("  '-selfcheck': compares the optimal objective with sp_vars_size being 400 and "
	 "de_vars_size being 100 (these two exact values must be passed as arguments). [optional]\n");
  printf("  'eq_ineq_combined_nlp': 0 or 1, specifying whether the NLP formulation with split "
	 "constraints should be used (0) or not (1) [default 0, optional]\n");
}


int main(int argc, char **argv)
{
  int rank=0;
#ifdef HIOP_USE_MPI
  MPI_Init(&argc, &argv);
  int comm_size;
  int ierr = MPI_Comm_size(MPI_COMM_WORLD, &comm_size); assert(MPI_SUCCESS==ierr);
  //int ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank); assert(MPI_SUCCESS==ierr);
  if(comm_size != 1) {
    printf("[error] driver detected more than one rank but the driver should be run "
	   "in serial only; will exit\n");
    MPI_Finalize();
    return 1;
  }
#endif

#ifdef HIOP_USE_MAGMA
  magma_init();
#endif

  // Set memory space where to create models
  std::string mem_space = "um";

  bool selfCheck, one_call_cons;
  long long n_sp, n_de;
  if(!parse_arguments(argc, argv, selfCheck, n_sp, n_de, one_call_cons)) {
    usage(argv[0]);
    return 1;
  }

  double obj_value=-1e+20;
  hiopSolveStatus status;

  //user's NLP -> implementation of hiop::hiopInterfaceMDS
  Ex4* my_nlp;
  if(one_call_cons)
  {
    my_nlp = new Ex4OneCallCons(n_sp, n_de, mem_space);
  }
  else
  {
    my_nlp = new Ex4(n_sp, n_de, mem_space);
  }

  hiopNlpMDS nlp(*my_nlp);

  nlp.options->SetStringValue("duals_update_type", "linear");
  nlp.options->SetStringValue("duals_init", "zero");

  nlp.options->SetStringValue("fixed_var", "relax");
  nlp.options->SetStringValue("Hessian", "analytical_exact");
  nlp.options->SetStringValue("KKTLinsys", "xdycyd");
  nlp.options->SetStringValue("compute_mode", "gpu");
  nlp.options->SetStringValue("mem_space", mem_space.c_str());

  nlp.options->SetIntegerValue("verbosity_level", 3);
  nlp.options->SetNumericValue("mu0", 1e-1);
  nlp.options->SetNumericValue("tolerance", 1e-5);

  hiopAlgFilterIPMNewton solver(&nlp);
  status = solver.run();
  obj_value = solver.getObjective();
  
  if(status<0) {
    if(rank==0)
      printf("solver returned negative solve status: %d (with objective is %18.12e)\n", status, obj_value);
    return -1;
  }

#if 0
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // Reoptimize
  // -----------
  // 1. get solution from previous solve
  // 2. give it to the (user's) nlp, which will provide HiOp a full primal-dual restart via
  // 'get_starting_point' callback
  // Normally, the user would also change her nlp between steps 1 and 2 above, for example, different
  // bounds on variables or on inequalities
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  long long n_vars, n_cons;
  my_nlp->get_prob_sizes(n_vars, n_cons);

  double* x       = hiop::LinearAlgebraFactory::createRawArray(n_vars);
  double* zl      = hiop::LinearAlgebraFactory::createRawArray(n_vars);
  double* zu      = hiop::LinearAlgebraFactory::createRawArray(n_vars);
  double* lambdas = hiop::LinearAlgebraFactory::createRawArray(n_cons);

  solver.getSolution(x);
  solver.getDualSolutions(zl, zu, lambdas);

  my_nlp->set_solution_primal(x);
  my_nlp->set_solution_duals(zl, zu, lambdas);

  //
  // set options for solver re-optimization
  //
  
  //less agressive log-barrier parameter is always a safe bet
  nlp.options->SetNumericValue("mu0", 1e-6);
  nlp.options->SetNumericValue("tolerance", 1e-8);

  //nlp.options->SetIntegerValue("verbosity_level", 7);

  //nlp.options->SetNumericValue("kappa1", 1e-15);
  //nlp.options->SetNumericValue("kappa2", 1e-15);
  
  //solve
  status = solver.run();
  obj_value = solver.getObjective();
  
  if(status<0) {
    if(rank==0)
      printf("solver returned negative solve status: %d (with objective is %18.12e)\n", status, obj_value);
    return -1;
  }

  if(selfCheck) {
    if(fabs(obj_value-(-4.999509728895e+01))>1e-6) {
      printf("selfcheck: objective mismatch for Ex4 MDS problem with 400 sparse variables and 100 "
	     "dense variables did. BTW, obj=%18.12e was returned by HiOp.\n", obj_value);
      return -1;
    }
  } else {
    if(rank==0) {
      printf("Optimal objective: %22.14e. Solver status: %d\n", obj_value, status);
    }
  }
#endif
  
  delete my_nlp;
  
#ifdef HIOP_USE_MAGMA
  magma_finalize();
#endif
#ifdef HIOP_USE_MPI
  MPI_Finalize();
#endif

  return 0;
}
