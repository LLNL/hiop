#include "NlpDenseConsEx4.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"

#include <cstdlib>
#include <string>

using namespace hiop;

static bool self_check(size_type n, double obj_value);
static bool self_check_uncon(size_type n, double obj_value);

int main(int argc, char **argv)
{
  int rank=0;
#ifdef HIOP_USE_MPI
  MPI_Init(&argc, &argv);
  int ierr = MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  assert(MPI_SUCCESS==ierr);
  //if(0==rank) printf("Support for MPI is enabled\n");
#endif
  bool selfCheck;
  bool unconstrained;
  size_type n;

  DenseConsEx4 nlp_interface;
  //if(rank==0) printf("interface created\n");
  hiopNlpDenseConstraints nlp(nlp_interface);
  //if(rank==0) printf("nlp formulation created\n");

  nlp.options->SetStringValue("duals_update_type", "linear");
  nlp.options->SetStringValue("compute_mode", "cpu");
  nlp.options->SetNumericValue("mu0", 0.1);

  hiopAlgFilterIPM solver(&nlp);
  hiopSolveStatus status = solver.run();

  double obj_value = solver.getObjective();
  
  if(status<0) {
    if(rank==0) printf("solver returned negative solve status: %d (with objective is %18.12e)\n", status, obj_value);
    return -1;
  }

  //this is used for "regression" testing when the driver is called with -selfcheck
  if(selfCheck) {
    if(!unconstrained) {
      if(!self_check(n, obj_value)) {
        return -1;
      }
    } else {
      if(!self_check_uncon(n, obj_value)) {
        return -1;
      }  
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
  const size_type n_saved[] = {500, 5000, 50000}; 
  const double objval_saved[] = {1.56251020819349e-02, 1.56251019995139e-02, 1.56251028980352e-02};

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

static bool self_check_uncon(size_type n, double objval)
{
#define num_n_saved 3 //keep this is sync with n_saved and objval_saved
  const size_type n_saved[] = {500, 5000, 50000}; 
  const double objval_saved[] = {1.56250004019985e-02, 1.56250035348275e-02, 1.56250304912460e-02};

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