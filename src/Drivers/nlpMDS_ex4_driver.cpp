#include "nlpMDSForm_ex4.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"

#ifdef HIOP_USE_MAGMA
#include "magma_v2.h"
#endif

#include <cstdlib>
#include <string>

using namespace hiop;

static bool self_check(long long n, double obj_value);

static bool parse_arguments(int argc, char **argv, long long& n, bool& self_check)
{
  self_check=false; n = 40;
  switch(argc) {
  case 1:
    //no arguments
    return true;
    break;
  case 3: //2 arguments
    {
      if(std::string(argv[2]) == "-selfcheck")
	self_check=true;
      else {
	n = std::atoi(argv[2]);
	if(n<=0) return false;
      }
    }
  case 2: //1 argument
    {
      if(std::string(argv[1]) == "-selfcheck")
	self_check=true;
      else {
	n = std::atoi(argv[1]);
	if(n<=0) return false;
      }
    }
    break;
  default: 
    return false; //3 or more arguments
  }

  return true;
};

static void usage(const char* exeName)
{
  printf("hiOp driver %s that solves a synthetic problem of variable size in the mixed dense-sparse formulation.\n", exeName);
  printf("Usage: \n");
  printf("  '$ %s sp_vars_size -selfcheck'\n", exeName);
  printf("Arguments:\n");
  printf("  'sp_vars_size': # of sparse variables (# of dense variables will be a quarter of this value) [optional, default is 50k]\n");
  printf("  '-selfcheck': compares the optimal objective with a previously saved value for the problem specified by 'problem_size'. [optional]\n");
}


int main(int argc, char **argv)
{

#ifdef HIOP_USE_MPI
  MPI_Init(&argc, &argv);
  int comm_size;
  int ierr = MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  assert(MPI_SUCCESS==ierr);
  if(comm_size != 1) {
    printf("[error] driver detected more than one rank but the driver should be run in serial only; will exit\n");
    MPI_Finalize();
    return 1;
  }
#endif

#ifdef HIOP_USE_MAGMA
  magma_init();
#endif

  bool selfCheck; long long n;
  if(!parse_arguments(argc, argv, n, selfCheck)) { usage(argv[0]); return 1;}

  double obj_value=-1e+20;
  hiopSolveStatus status;

  Ex4 nlp_interface(8000);

  hiopNlpMDS nlp(nlp_interface);

  nlp.options->SetStringValue("dualsUpdateType", "linear");
  nlp.options->SetStringValue("dualsInitialization", "zero");

  nlp.options->SetStringValue("Hessian", "analytical_exact");
  nlp.options->SetStringValue("KKTLinsys", "xdycyd");
  nlp.options->SetStringValue("compute_mode", "hybrid");

  nlp.options->SetIntegerValue("verbosity_level", 3);
  nlp.options->SetNumericValue("mu0", 1e-1);
  hiopAlgFilterIPMNewton solver(&nlp);
  status = solver.run();
  obj_value = solver.getObjective();

  // if(status<0) {
  //   if(rank==0) printf("solver returned negative solve status: %d (with objective is %18.12e)\n", status, obj_value);
  //   return -1;
  // }

  // //this is used for "regression" testing when the driver is in '-selfcheck' mode
  // if(selfCheck) {
  //   if(!self_check(n, obj_value))
  //     return -1;
  // } else {
  //   if(rank==0) {
  //     printf("Optimal objective: %22.14e. Solver status: %d\n", obj_value, status);
  //   }
  // }
#ifdef HIOP_USE_MAGMA
magma_finalize();
#endif
#ifdef HIOP_USE_MPI
  MPI_Finalize();
#endif

  return 0;
}


static bool self_check(long long n, double objval)
{
#define num_n_saved 3 //keep this is sync with n_saved and objval_saved
  const long long n_saved[] = {500, 5000, 50000}; 
  const double objval_saved[] = {2.05788282767327e+00, 2.02870382737020e+01, 2.02578703828247e+02};

#define relerr 1e-6
  bool found=false;
  for(int it=0; it<num_n_saved; it++) {
    if(n_saved[it]==n) {
      found=true;
      if(fabs( (objval_saved[it]-objval)/(1+objval_saved[it])) > relerr) {
	printf("selfcheck failure. Objective (%18.12e) does not agree (%d digits) with the saved value (%18.12e) for n=%lld.\n",
	       objval, -(int)log10(relerr), objval_saved[it], n);
	return false;
      } else {
	printf("selfcheck success (%d digits)\n",  -(int)log10(relerr));
      }
      break;
    }
  }

  if(!found) {
    printf("selfcheck: driver does not have the objective for n=%lld saved. BTW, obj=%18.12e was obtained for this n.\n", n, objval);
    return false;
  }

  return true;
}
