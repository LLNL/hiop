//the problem to be solved
#include "nlpPriDec_ex8.hpp"
//the solver
#include "hiopAlgPrimalDecomp.hpp"

#ifdef HIOP_USE_MAGMA
#include "magma_v2.h"
#endif

#include <cstdlib>
#include <string>

#ifdef HIOP_USE_MPI
#include "mpi.h"

#else
#define MPI_Comm int
#define MPI_COMM_WORLD 0
#endif


/**
 * Driver for example 8 that illustrates the use of hiop::hiopAlgPrimalDecomposition 
 * 
 * @note This example is built only when HIOP_USE_MPI is enabled during cmake build
 * and require at least two MPI ranks in MPI_COMM_WORLD.
 *
 */

//
//todo: add -selfcheck option (see other drivers) and add the driver to cmake tests
//

int main(int argc, char **argv)
{
  int rank=0;
#ifdef HIOP_USE_MPI
  MPI_Init(&argc, &argv);
  int comm_size;
  int ierr = MPI_Comm_size(MPI_COMM_WORLD, &comm_size); assert(MPI_SUCCESS==ierr);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank); assert(MPI_SUCCESS==ierr);
#endif

#ifdef HIOP_USE_MAGMA
  magma_init();
#endif
  int nx=20;
  int nS=100;
   
  //printf("here\n");
  PriDecMasterProblemEx8 pridec_problem(nx, nS);
  //printf("total ranks %d\n",comm_size);
  hiop::hiopAlgPrimalDecomposition pridec_solver(&pridec_problem, MPI_COMM_WORLD);

  auto status = pridec_solver.run();

  if(status!=Solve_Success){
    if(rank==0)
      printf("Solve was NOT successfull.");
  }else{
    if(rank==0)
      printf("Solve was successfull. Optimal value: %12.5e\n",
             pridec_solver.getObjective());
  }

#ifdef HIOP_USE_MAGMA
  magma_finalize();
#endif
#ifdef HIOP_USE_MPI
  MPI_Finalize();
#endif

  printf("Returned successfully from driver! Rank=%d\n", rank);;
  return 0;
}
