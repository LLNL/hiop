#pragma once

#include "vectorTests.hpp"

namespace hiopTest {

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
