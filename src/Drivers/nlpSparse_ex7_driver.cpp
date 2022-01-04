#include "nlpSparse_ex7.hpp"
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
                            bool& inertia_free)
{
  self_check = false;
  n = 3;
  inertia_free = false;
  switch(argc) {
  case 1:
    //no arguments
    return true;
    break;
  case 4: //3 arguments
    {
      if(std::string(argv[3]) == "-selfcheck") {
        self_check = true;    
      } else if(std::string(argv[3]) == "-inertiafree") {
        inertia_free = true;
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
      } else {
        n = std::atoi(argv[1]);
        if(n<=0) {
          return false;
        }
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
  printf("hiOp driver %s that solves a synthetic convex problem of variable size.\n", exeName);
  printf("Usage: \n");
  printf("  '$ %s problem_size -inertiafree -selfcheck'\n", exeName);
  printf("Arguments:\n");
  printf("  'problem_size': number of decision variables [optional, default is 50]\n");
  printf("  '-inertiafree': indicate if inertia free approach should be used [optional]\n");
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
  bool selfCheck; size_type n;
  bool inertia_free; 
  if(!parse_arguments(argc, argv, n, selfCheck, inertia_free)) { 
    usage(argv[0]);
    return 1;
  }

  bool convex_obj = false;
  bool rankdefic_Jac_eq = true;
  bool rankdefic_Jac_ineq = true;
  double scal_neg_obj = 0.1;

  //first test
  {
    Ex7 nlp_interface(n,convex_obj,rankdefic_Jac_eq,rankdefic_Jac_ineq, scal_neg_obj);
    hiopNlpSparse nlp(nlp_interface);
    nlp.options->SetStringValue("compute_mode", "cpu");
    nlp.options->SetStringValue("KKTLinsys", "xdycyd");
    if(inertia_free) {
      nlp.options->SetStringValue("fact_acceptor", "inertia_free");
    }
    hiopAlgFilterIPMNewton solver(&nlp);
    hiopSolveStatus status = solver.run();
    
    double obj_value = solver.getObjective();
    
    if(status<0) {
      if(rank==0) {
        printf("solver returned negative solve status: %d (with objective is %18.12e)\n", status, obj_value);
      }
      return -1;
    }

    //this is used for "regression" testing when the driver is called with -selfcheck
    if(selfCheck) {
      if(!self_check(n, obj_value, inertia_free))
        return -1;
    } else {
      if(rank==0) {
        printf("Optimal objective: %22.14e. Solver status: %d\n", obj_value, status);
      }
    }
  }
  
  //
  //same as above but with equalities relaxed as two-sided inequalities and using condensed linear system
  //
#if defined(HIOP_USE_CUDA) || defined(HIOP_USE_COINHSL)
  {
    Ex7 nlp_interface(n,convex_obj, rankdefic_Jac_eq, rankdefic_Jac_ineq, scal_neg_obj);
    hiopNlpSparseIneq nlp(nlp_interface);
#ifdef HIOP_USE_CUDA
    nlp.options->SetStringValue("compute_mode", "hybrid");
#else //HIOP_USE_COINHSL
    //compute mode cpu will use MA57 by default
    nlp.options->SetStringValue("compute_mode", "cpu");
#endif

    nlp.options->SetStringValue("KKTLinsys", "condensed");
    //disregard inertia_free command parameter since it is not yet supported
    //if(inertia_free) {
    //  nlp.options->SetStringValue("fact_acceptor", "inertia_free");
    //}

    hiopAlgFilterIPMNewton solver(&nlp);
    hiopSolveStatus status = solver.run();

    double obj_value = solver.getObjective();

    if(status<0) {
      if(rank==0) {
        printf("solver returned negative solve status with hiopNlpSparseIneq: %d (obj. is %18.12e)\n",
               status,
               obj_value);
      }
      return -1;
    }

    //this is used for "regression" testing when the driver is called with -selfcheck
    if(selfCheck) {
      if(!self_check(n, obj_value, inertia_free))
        return -1;
    } else {
      if(rank==0) {
        printf("Optimal objective: %22.14e. Solver status: %d\n", obj_value, status);
      }
    }
  }
#endif //HIOP_USE_CUDA
  
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
