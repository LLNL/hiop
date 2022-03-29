#include "nlpSparse_ex6.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"

#include <cstdlib>
#include <string>

using namespace hiop;

static bool self_check(size_type n, double obj_value);

static bool parse_arguments(int argc,
                            char **argv,
                            size_type& n,
                            double &scal,
                            bool& self_check,
                            bool& use_pardiso,
                            bool& use_cusolver,
                            bool& force_fr)
{
  self_check = false;
  use_pardiso = false;
  force_fr = false;
  n = 3;
  scal = 1.0;
  switch(argc) {
  case 1:
    //no arguments
    return true;
    break;
  case 6: //5 arguments
    {
      if(std::string(argv[5]) == "-selfcheck") {
        self_check = true;
      }
    }
  case 5: //4 arguments
    {
      if(std::string(argv[4]) == "-fr") {
        force_fr = true;
      } else if(std::string(argv[4]) == "-selfcheck") {
        self_check = true;
      } else if(std::string(argv[4]) == "-pardiso") {
        use_pardiso = true;
      } else if(std::string(argv[4]) == "-cusolver") {
        use_cusolver = true;
      }
    }
  case 4: //3 arguments
    {
      if(std::string(argv[3]) == "-fr") {
        force_fr = true;
      } else if(std::string(argv[3]) == "-selfcheck") {
        self_check = true;
      } else if(std::string(argv[3]) == "-pardiso") {
        use_pardiso = true;
      } else if(std::string(argv[3]) == "-cusolver") {
        use_cusolver = true;
      }
    }
  case 3: //2 arguments
    {
      if(std::string(argv[2]) == "-fr") {
        force_fr = true;
      } else if(std::string(argv[2]) == "-selfcheck") {
        self_check = true;
      } else if(std::string(argv[2]) == "-pardiso") {
        use_pardiso = true;
      } else if(std::string(argv[2]) == "-cusolver") {
        use_cusolver = true;
      } else {
        scal = std::atof(argv[2]); 
      }
    }
  case 2: //1 argument
    {
      n = std::atoi(argv[1]);
      if(n <= 0) {
        return false;
      }
    }
    break;
  default:
    return false; //6 or more arguments
  }
  if(self_check) {
    scal = 1.0;
  }

  if(use_cusolver && use_pardiso) {
    printf("Selected both, cuSOLVER and Pardiso. ");
    printf("You can select only one linear solver.\n\n");
    return false;
  }

// If Pardiso is not available de-select it.
#ifndef HIOP_USE_PARDISO
  if(use_pardiso) {
    printf("HiOp not built with Pardiso, using default linear solver ...\n");
    use_pardiso = false;
  }
#endif

// If HiOp is built without CUDA de-select cuSOLVER.
#ifndef HIOP_USE_CUDA
  if(use_cusolver) {
    printf("HiOp not built with CUDA support, using CPU linear solver ...\n");
    use_cusolver = false;
  }
#endif

  return true;
};

static void usage(const char* exeName)
{
  printf("hiOp driver %s that solves a synthetic convex problem of variable size.\n", exeName);
  printf("Usage: \n");
  printf("  '$ %s problem_size scal_fact -selfcheck'\n", exeName);
  printf("Arguments:\n");
  printf("  'problem_size': number of decision variables [optional, default is 50]\n");
  printf("  'scal_fact': scaling factor used for objective function and constraints [optional, "
         "default is 1.0]\n");
  printf("  '-pardiso' or '-cusolver': use Pardiso or cuSOLVER "
         "as the linear solver [optional]\n");
  printf("  '-cusolver': use cuSOLVER as the linear solver [optional]\n");
  printf("  '-fr': force to reset feasibility in the 1st iteration [optional]\n");
  printf("  '-selfcheck': compares the optimal objective with a previously saved value for the "
         "problem specified by 'problem_size'. [optional]\n");
}


int main(int argc, char **argv)
{
  int rank=0;
#ifdef HIOP_USE_MPI
  MPI_Init(&argc, &argv);
  int comm_size;
  int ierr = MPI_Comm_size(MPI_COMM_WORLD, &comm_size); assert(MPI_SUCCESS==ierr);
  if(comm_size != 1) {
    printf("[error] driver detected more than one rank but the driver should be run "
           "in serial only; will exit\n");
    MPI_Finalize();
    return 1;
  }
#endif
  bool selfCheck = false;
  bool use_pardiso = false;
  bool use_cusolver = false;
  bool force_fr = false;
  size_type n;
  double scal;

  if(!parse_arguments(argc, argv, n, scal, selfCheck, use_pardiso, use_cusolver, force_fr)) {
    usage(argv[0]);
#ifdef HIOP_USE_MPI
    MPI_Finalize();
#endif
    return 1;
  }

  Ex6 nlp_interface(n, scal);
  hiopNlpSparse nlp(nlp_interface); 
  nlp.options->SetStringValue("Hessian", "analytical_exact");
  
  nlp.options->SetStringValue("duals_update_type", "linear"); 
  
  nlp.options->SetStringValue("compute_mode", "cpu");
  nlp.options->SetStringValue("KKTLinsys", "xdycyd");
  //nlp.options->SetStringValue("write_kkt", "yes");

  nlp.options->SetNumericValue("mu0", 0.1);
  nlp.options->SetStringValue("options_file_fr_prob", "hiop_fr_ci.options");

  if(use_pardiso) {
    nlp.options->SetStringValue("linear_solver_sparse", "pardiso");
  }
  if(use_cusolver) {
    nlp.options->SetStringValue("linsol_mode", "speculative");
    nlp.options->SetStringValue("linear_solver_sparse", "cusolver");
    nlp.options->SetStringValue("compute_mode", "hybrid");
    nlp.options->SetStringValue("fact_acceptor", "inertia_free");
  }
  if(force_fr) {
    nlp.options->SetStringValue("force_resto", "yes");
  }

  hiopAlgFilterIPMNewton solver(&nlp);
  hiopSolveStatus status = solver.run();

  double obj_value = solver.getObjective();
  if(status<0) {
    if(rank==0) {
      printf("solver returned negative solve status: %d (obj. is %18.12e)\n",
             status,
             obj_value);
    }
#ifdef HIOP_USE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  //this is used for "regression" testing when the driver is called with -selfcheck
  if(selfCheck) {
    if(!self_check(n, obj_value)) {
#ifdef HIOP_USE_MPI
      MPI_Finalize();
#endif
      return -1;
    }
  } else {
    if(rank==0) {
      printf("Optimal objective: %22.14e. Solver status: %d\n", obj_value, status);
    }
  }

#ifdef HIOP_USE_MPI
  MPI_Finalize();
#endif


  return 0;
}


static bool self_check(size_type n, double objval)
{
#define num_n_saved 3 //keep this is sync with n_saved and objval_saved
  const size_type n_saved[] = {50, 500, 5000};
  const double objval_saved[] = {1.10351564683176e-01, 1.10351566513480e-01, 1.10351578644469e-01};

#define relerr 1e-6
  bool found=false;
  for(int it=0; it<num_n_saved; it++) {
    if(n_saved[it]==n) {
      found=true;
      if(fabs( (objval_saved[it]-objval)/(1+objval_saved[it])) > relerr) {
        printf("selfcheck failure. Objective (%18.12e) does not agree (%d digits) with the saved value (%18.12e) for n=%d.\n",
               objval, -(int)log10(relerr), objval_saved[it], n);
        return false;
      } else {
        printf("selfcheck success (%d digits)\n",  -(int)log10(relerr));
      }
      break;
    }
  }

  if(!found) {
    printf("selfcheck: driver does not have the objective for n=%d saved. BTW, obj=%18.12e was obtained for this n.\n", n, objval);
    return false;
  }

  return true;
}
