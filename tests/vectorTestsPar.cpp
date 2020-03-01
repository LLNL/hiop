#include <hiopVector.hpp>
#include "vectorTestsPar.hpp"

namespace hiopTest {


void VectorTestsPar::setElement(hiop::hiopVector* x, int i, double value)
{
    hiop::hiopVectorPar* xvec = dynamic_cast<hiop::hiopVectorPar*>(x);
    double* xdat = xvec->local_data();
    xdat[i] = value;
}

double VectorTestsPar::getElement(const hiop::hiopVector* x, int i)
{
    const hiop::hiopVectorPar* xvec = dynamic_cast<const hiop::hiopVectorPar*>(x);
    return xvec->local_data_const()[i];
}

} // namespace hiopTest
