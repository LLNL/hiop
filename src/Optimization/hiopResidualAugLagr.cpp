#include "hiopResidualAugLagr.hpp"

namespace hiop
{

/* residual printing function - calls hiopVector::print 
 * prints up to max_elems (by default all), on rank 'rank' (by default on all) */
virtual void hiopResidualAugLagr::print(FILE*, const char* msg=NULL, int max_elems=-1, int rank=-1) const
{
}

}
