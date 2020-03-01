#pragma once

#include "matrixTests.hpp"

namespace hiop::tests {

class MatrixTestsDense : public MatrixTests
{
public:
    MatrixTestsDense(){}
    virtual ~MatrixTestsDense(){}

private:
    virtual void setElement(hiop::hiopMatrix* a, int i, int j, double val);
    virtual double getElement(hiop::hiopMatrix* a, int i, int j);
    virtual int getNumLocRows(hiop::hiopMatrix* a);
    virtual int getNumLocCols(hiop::hiopMatrix* a);

};

} // namespace hiop::tests
