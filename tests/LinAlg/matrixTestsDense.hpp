#pragma once

#include "matrixTests.hpp"

namespace hiop::tests {

class MatrixTestsDense : public MatrixTests
{
public:
    MatrixTestsDense(){}
    virtual ~MatrixTestsDense(){}

private:
    virtual void setElement(hiop::hiopMatrix* a, local_ordinal_type i, local_ordinal_type j, real_type val);
    virtual real_type getElement(hiop::hiopMatrix* a, local_ordinal_type i, local_ordinal_type j);
    virtual local_ordinal_type getNumLocRows(hiop::hiopMatrix* a);
    virtual local_ordinal_type getNumLocCols(hiop::hiopMatrix* a);

};

} // namespace hiop::tests
