#pragma once

#include "vectorTests.hpp"

namespace hiop::tests {

/**
 * @brief Utilities for testing hiopVectorPar class
 *
 * @todo In addition to set and get element ass set and get buffer methods.
 *
 */
class VectorTestsPar : public VectorTests
{
public:
    VectorTestsPar(){}
    virtual ~VectorTestsPar(){}

private:
    virtual void setLocalElement(hiop::hiopVector* x, local_ordinal_type i, real_type value);
    virtual real_type getLocalElement(const hiop::hiopVector* x, local_ordinal_type i);
    virtual local_ordinal_type getLocalSize(const hiop::hiopVector* x);
    virtual real_type* getLocalData(hiop::hiopVector* x);
    virtual int verifyAnswer(hiop::hiopVector* x, real_type answer);
    virtual int verifyAnswer(
        hiop::hiopVector* x,
        std::function<real_type(local_ordinal_type)> expect);
    virtual bool reduceReturn(int failures, hiop::hiopVector* x);

#ifdef HIOP_USE_MPI
    MPI_Comm getMPIComm(hiop::hiopVector* x);
#endif
};

} // namespace hiopTest
