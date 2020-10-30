
//the solver
#include "hiopAlgPrimalDecomp.hpp"
//the problem to be solved
#include "nlpPriDec_ex8.hpp"


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

  PriDecMasterProblemEx8 pridec_problem(12, 20);

  hiop::hiopAlgPrimalDecomposition pridec_solver(&pridec_problem, MPI_COMM_WORLD);

  auto status = pridec_solver.run();

  if(status!=Solve_Success) {
    if(rank==0)
      printf("Solve was NOT successfull.");
  } else {
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
