#include "NlpSparseEx2.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"

#include <cstdlib>
#include <string>

using namespace hiop;

static bool self_check(size_type n, double obj_value, const bool inertia_free);

static bool parse_arguments(int argc,
                            char **argv,
                            size_type& n,
                            bool& self_check,
                            bool& inertia_free,
                            bool& use_cusolver)
{
  self_check = false;
  n = 3;
  inertia_free = false;
  use_cusolver = false;
  switch(argc) {
  case 1:
    //no arguments
    return true;
    break;
  case 5: //4 arguments
    {
      if(std::string(argv[4]) == "-selfcheck") {
        self_check = true;    
      } else if(std::string(argv[4]) == "-inertiafree") {
        inertia_free = true;
      } else if(std::string(argv[4]) == "-cusolver") {
        use_cusolver = true;
      } else {
        n = std::atoi(argv[4]);
        if(n<=0) {
          return false;
        }
      }
    }
  case 4: //3 arguments
    {
      if(std::string(argv[3]) == "-selfcheck") {
        self_check = true;    
      } else if(std::string(argv[3]) == "-inertiafree") {
        inertia_free = true;
      } else if(std::string(argv[3]) == "-cusolver") {
        use_cusolver = true;
      } else {
        n = std::atoi(argv[3]);
        if(n<=0) {
          return false;
        }
      }
    }
    case 3: //2 arguments
    {
      if(std::string(argv[2]) == "-selfcheck") {
        self_check = true;    
      } else if(std::string(argv[2]) == "-inertiafree") {
        inertia_free = true;
      } else if(std::string(argv[2]) == "-cusolver") {
        use_cusolver = true;
      } else {
        n = std::atoi(argv[2]);
        if(n<=0) {
          return false;
        }
      }
    }
  case 2: //1 argument
    {
      if(std::string(argv[1]) == "-selfcheck") {
        self_check = true;    
      } else if(std::string(argv[1]) == "-inertiafree") {
        inertia_free = true;
      } else if(std::string(argv[1]) == "-cusolver") {
        use_cusolver = true;
      } else {
        n = std::atoi(argv[1]);
        if(n<=0) {
          return false;
        }
      }
    }
    break;
  default:
    return false; // 4 or more arguments
  }

  if(use_cusolver && !(inertia_free)) {
    inertia_free = true;
    printf("LU solver from cuSOLVER library requires inertia free approach. ");
    printf("Enabling now ...\n");
  }

#ifndef HIOP_USE_CUDA
  if(use_cusolver) {
    printf("HiOp built without CUDA support. ");
    printf("Using default instead of cuSOLVER ...\n");
    use_cusolver = false;
  }
#endif

  return true;
};

static void usage(const char* exeName)
{
  printf("hiOp driver %s that solves a synthetic convex problem of variable size.\n", exeName);
  printf("Usage: \n");
  printf("  '$ %s problem_size -inertiafree -selfcheck'\n", exeName);
  printf("Arguments:\n");
  printf("  'problem_size': number of decision variables [optional, default is 50]\n");
  printf("  '-inertiafree': indicate if inertia free approach should be used [optional]\n");
  printf("  '-selfcheck': compares the optimal objective with a previously saved value for the "
         "problem specified by 'problem_size'. [optional]\n");
  printf("  '-cusolver': use cuSOLVER linear solver [optional]\n");
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
  size_type n = 50;
  bool inertia_free = false;
  bool use_cusolver = false;
  if(!parse_arguments(argc, argv, n, selfCheck, inertia_free, use_cusolver)) { 
    usage(argv[0]);
#ifdef HIOP_USE_MPI
    MPI_Finalize();
#endif
    return 1;
  }

  bool convex_obj = false;
  bool rankdefic_Jac_eq = true;
  bool rankdefic_Jac_ineq = true;
  double scal_neg_obj = 0.1;

  //first test
  {
    SparseEx2 nlp_interface(n,convex_obj,rankdefic_Jac_eq,rankdefic_Jac_ineq, scal_neg_obj);
    hiopNlpSparse nlp(nlp_interface);
    nlp.options->SetStringValue("compute_mode", "cpu");
    nlp.options->SetStringValue("KKTLinsys", "xdycyd");
    //lsq initialization of the duals fails for this example since the Jacobian is rank deficient
    //use zero initialization
    nlp.options->SetStringValue("duals_init", "zero");
    if(inertia_free) {
      nlp.options->SetStringValue("fact_acceptor", "inertia_free");
    }
    if(use_cusolver) {
      nlp.options->SetStringValue("duals_init", "zero");
      nlp.options->SetStringValue("linsol_mode", "speculative");
      nlp.options->SetStringValue("linear_solver_sparse", "cusolver-lu");
      nlp.options->SetStringValue("compute_mode", "hybrid");
    }
    hiopAlgFilterIPMNewton solver(&nlp);
    hiopSolveStatus status = solver.run();
    
    double obj_value = solver.getObjective();
    
    if(status<0) {
      if(rank==0) {
        printf("solver returned negative solve status: %d (with objective is %18.12e)\n", status, obj_value);
      }
#ifdef HIOP_USE_MPI
      MPI_Finalize();
#endif
      return -1;
    }

    //this is used for "regression" testing when the driver is called with -selfcheck
    if(selfCheck) {
      if(!self_check(n, obj_value, inertia_free)) {
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
  }
  
  //
  //same as above but with equalities relaxed as two-sided inequalities and using condensed linear system
  //
#ifdef HIOP_USE_COINHSL
  {
    SparseEx2 nlp_interface(n,convex_obj, rankdefic_Jac_eq, rankdefic_Jac_ineq, scal_neg_obj);
    hiopNlpSparseIneq nlp(nlp_interface);
    //compute mode cpu will use MA57 by default
    nlp.options->SetStringValue("KKTLinsys", "condensed");
    nlp.options->SetStringValue("compute_mode", "cpu");
    nlp.options->SetStringValue("linsol_mode", "speculative");
    nlp.options->SetStringValue("duals_init", "zero");
    if(use_cusolver) {
      nlp.options->SetStringValue("compute_mode", "hybrid");
    }

    hiopAlgFilterIPMNewton solver(&nlp);
    hiopSolveStatus status = solver.run();

    double obj_value = solver.getObjective();

    if(status<0) {
      if(rank==0) {
        printf("solver returned negative solve status with hiopNlpSparseIneq: %d (obj. is %18.12e)\n",
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
      if(!self_check(n, obj_value, inertia_free)) {
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
  }
#endif //HIOP_USE_COINHSL
  
#ifdef HIOP_USE_MPI
  MPI_Finalize();
#endif

  return 0;
}


static bool self_check(size_type n, double objval, const bool inertia_free)
{
#define num_n_saved 3 //keep this is sync with n_saved and objval_saved
  const size_type n_saved[] = {50, 500, 10000};
  const double objval_saved[] = { 8.7754974e+00,  6.4322371e+01,  1.2369786e+03};

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
