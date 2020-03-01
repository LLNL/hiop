#include <hiopVector.hpp>
#include "vectorTestsPar.hpp"

namespace hiop::tests {

/// Method to set vector _x_ element _i_ to _value_.
/// First need to retrieve hiopVectorPar from the abstract interface
void VectorTestsPar::setElement(hiop::hiopVector* x, int i, double value)
{
    hiop::hiopVectorPar* xvec = dynamic_cast<hiop::hiopVectorPar*>(x);
    double* xdat = xvec->local_data();
    xdat[i] = value;
}

/// Returns element _i_ of vector _x_.
/// First need to retrieve hiopVectorPar from the abstract interface
double VectorTestsPar::getElement(const hiop::hiopVector* x, int i)
{
    const hiop::hiopVectorPar* xvec = dynamic_cast<const hiop::hiopVectorPar*>(x);
    return xvec->local_data_const()[i];
}

} // namespace hiop::tests
