#pragma once

#include <hiopMatrix.hpp>

namespace hiop::tests {

class MatrixTests
{
public:
    MatrixTests(){}
    virtual ~MatrixTests(){}

    int matrixNumRows(hiop::hiopMatrix& A, long long M)
    {
        return A.m() == M ? 0 : 1;
    }

    int matrixNumCols(hiop::hiopMatrix& A, long long N)
    {
        return A.n() == N ? 0 : 1;
    }

    // Code here ...


};

} // namespace hiopTest
