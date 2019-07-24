#include "hiopResidualAugLagr.hpp"

namespace hiop
{

/* residual printing function - calls hiopVector::print 
 * prints up to max_elems (by default all), on rank 'rank' (by default on all) */
void hiopResidualAugLagr::print(FILE* f, const char* msg, int max_elems, int rank) const
{
}

}
