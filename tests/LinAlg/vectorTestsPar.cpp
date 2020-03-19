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

/// Returns pointer to local ector data
double* VectorTestsPar::getLocalData(hiop::hiopVector* x)
{
    hiop::hiopVectorPar* xvec = dynamic_cast<hiop::hiopVectorPar*>(x);
    return xvec->local_data();
}

/// Returns size of local data array for vector _x_
int VectorTestsPar::getLocalSize(const hiop::hiopVector* x)
{
    const hiop::hiopVectorPar* xvec = dynamic_cast<const hiop::hiopVectorPar*>(x);
    return static_cast<int>(xvec->get_local_size());
}

#ifdef HIOP_USE_MPI
/// Get communicator
MPI_Comm VectorTestsPar::getMPIComm(hiop::hiopVector* x)
{
    const hiop::hiopVectorPar* xvec = dynamic_cast<const hiop::hiopVectorPar*>(x);
    return xvec->get_mpi_comm();
}
#endif

/// If test fails on any rank set fail flag on all ranks
bool VectorTestsPar::reduceReturn(int failures, hiop::hiopVector* x)
{
    int fail = 0;

#ifdef HIOP_USE_MPI
    MPI_Allreduce(&failures, &fail, 1, MPI_INT, MPI_SUM, getMPIComm(x));
#else
    fail = failures;
#endif

    return (fail != 0);
}


/// Checks if _local_ vector elements are set to `answer`.
int VectorTestsPar::verifyAnswer(hiop::hiopVector* x, double answer)
{
    const int N = getLocalSize(x);
    const double* xdata = getLocalData(x);

    int local_fail = 0;
    for(int i=0; i<N; ++i)
        if(!isEqual(xdata[i], answer))
            ++local_fail;

    return local_fail;
}



} // namespace hiop::tests
