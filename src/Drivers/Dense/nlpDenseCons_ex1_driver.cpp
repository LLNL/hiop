#include "hiopNlpFormulation.hpp"
#include "hiopInterface.hpp"
#include "hiopAlgFilterIPM.hpp"

#include "nlpDenseCons_ex1.hpp"

#include <cstdlib>
#include <string>

using namespace hiop;

static bool self_check(size_type n, double obj_value);

static bool parse_arguments(int argc, char **argv, size_type& n, double& distortion_ratio, bool& self_check)
{
  n = 20000; distortion_ratio=1.; self_check=false; //default options

  switch(argc) {
  case 1:
    //no arguments
    return true;
    break;
  case 4: //3 arguments - -selfcheck expected
    {
      if(std::string(argv[3]) == "-selfcheck")
	self_check=true;
      else 
	return false;
    }
  case 3: //2 arguments: pick up distortion ratio here
    {
      distortion_ratio = atof(argv[2]);
    }
  case 2: //1 argument 
    {
      n = std::atoi(argv[1]);
    }
    break;
  default: 
    return false; //3 or more arguments
  }

  if(n<=0) return false;
  if(distortion_ratio<=1e-8 || distortion_ratio>1.) return false;
  return true;
};

static void usage(const char* exeName)
{
  printf("hiOp driver '%s' that solves a synthetic infinite dimensional problem of variable size. A 1D mesh is created by the example, and the size and the distortion of the mesh can be specified as options to this executable. The distortion of the mesh is the ratio of the smallest element and the largest element in the mesh.\n", exeName);
  printf("Usage: \n");
  printf("  '$ %s problem_size mesh_distortion_ratio -selfcheck'\n", exeName);
  printf("Arguments (specify in the order above): \n");
  printf("  'problem_size': number of decision variables [optional, default is 20k]\n");
  printf("  'dist_ratio': mesh distortion ratio, see above; a number in (0,1)  [optional, default 1.0]\n");
  printf("  '-selfcheck': compares the optimal objective with a previously saved value for the problem specified by 'problem_size'. [optional]\n");
}


int main(int argc, char **argv)
{
  int rank=0, numRanks=1;
#ifdef HIOP_USE_MPI
  int err;
  err = MPI_Init(&argc, &argv);                  assert(MPI_SUCCESS==err);
  err = MPI_Comm_rank(MPI_COMM_WORLD,&rank);     assert(MPI_SUCCESS==err);
  err = MPI_Comm_size(MPI_COMM_WORLD,&numRanks); assert(MPI_SUCCESS==err);
  if(0==rank) printf("Support for MPI is enabled\n");
#endif
  bool selfCheck; size_type mesh_size; double ratio;
  if(!parse_arguments(argc, argv, mesh_size, ratio, selfCheck)) { usage(argv[0]); return 1;}
  
  Ex1Interface problem(mesh_size, ratio);
  //if(rank==0) printf("interface created\n");
  hiop::hiopNlpDenseConstraints nlp(problem);
  //if(rank==0) printf("nlp formulation created\n");

  //nlp.options->SetIntegerValue("verbosity_level", 4);
  //nlp.options->SetNumericValue("tolerance", 1e-4);
  //nlp.options->SetStringValue("duals_init",  "zero");
  //nlp.options->SetIntegerValue("max_iter", 2);
  
  hiop::hiopAlgFilterIPM solver(&nlp);
  hiop::hiopSolveStatus status = solver.run();
  double objective = solver.getObjective();

  //this is used for testing when the driver is called with -selfcheck
  if(selfCheck) {
    if(!self_check(mesh_size, objective))
      return -1;
  } else {
    if(rank==0) {
      printf("Optimal objective: %22.14e. Solver status: %d. Number of iterations: %d\n", 
	     objective, status, solver.getNumIterations());
    }
  }

  if(0==rank) printf("Objective: %18.12e\n", objective);
#ifdef HIOP_USE_MPI
  MPI_Finalize();
#endif
  
  return 0;
}

static bool self_check(size_type n, double objval)
{
#define num_n_saved 3 //keep this is sync with n_saved and objval_saved
  const size_type n_saved[] = {500, 5000, 50000}; 
  const double objval_saved[] = {8.6156700e-2, 8.6156106e-02, 8.6161001e-02};

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
