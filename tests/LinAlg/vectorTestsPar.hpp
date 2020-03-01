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
};

} // namespace hiopTest
