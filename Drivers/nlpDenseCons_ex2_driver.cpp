#include "nlpDenseCons_ex2.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"



int main(int argc, char **argv)
{
  int rank=0;
#ifdef WITH_MPI
  MPI_Init(&argc, &argv);
  assert(MPI_SUCCESS==MPI_Comm_rank(MPI_COMM_WORLD,&rank));
  //assert(MPI_SUCCESS==MPI_Comm_size(MPI_COMM_WORLD,&numRanks));
  if(0==rank) printf("Support for MPI is enabled\n");
#endif

  Ex2 nlp_interface(7);
  if(rank==0) printf("interface created\n");
  hiopNlpDenseConstraints nlp(nlp_interface);
  if(rank==0) printf("nlp formulation created\n");

  hiopAlgFilterIPM solver(&nlp);
  solver.run();
#ifdef WITH_MPI
  MPI_Finalize();
#endif

  return 0;
}
