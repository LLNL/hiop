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
    virtual void setElement(hiop::hiopVector* x, int i, double value);
    virtual double getElement(const hiop::hiopVector* x, int i);
    virtual int getLocalSize(const hiop::hiopVector* x);
    virtual double* getLocalData(hiop::hiopVector* x);
    virtual int verifyAnswer(hiop::hiopVector* x, double answer);
    virtual bool reduceReturn(int failures, hiop::hiopVector* x);

#ifdef HIOP_USE_MPI
    MPI_Comm getMPIComm(hiop::hiopVector* x);
#endif
};

} // namespace hiopTest
