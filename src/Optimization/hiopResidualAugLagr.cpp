#include "hiopResidualAugLagr.hpp"

namespace hiop
{

/* residual printing function - calls hiopVector::print 
 * prints up to max_elems (by default all), on rank 'rank' (by default on all) */
void hiopResidualAugLagr::print(FILE* f, const char* msg, int max_elems, int rank) const
{
  if(NULL==msg) fprintf(f, "hiopResidual print\n");
  else fprintf(f, "%s\n", msg);

  _penaltyFcn->print(  f, "  penalty:", max_elems, rank); 
  _gradLagr->print(  f, " gradLagr:", max_elems, rank);   
  printf(" errors (optim/feasib) nlp    : %26.16e %25.16e\n", 
         _nrmInfOptim, _nrmInfFeasib);
  //printf(" errors (optim/feasib) barrier: %25.16e %25.16e\n", 
  //       nrmInf_bar_optim, nrmInf_bar_feasib);
}

}
