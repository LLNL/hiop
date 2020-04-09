#pragma once

#include <iostream>
#include <hiopMatrix.hpp>
#include "testBase.hpp"

namespace hiop::tests {

class MatrixTests : public TestBase
{
public:
    MatrixTests(){}
    virtual ~MatrixTests(){}

    global_ordinal_type matrixNumRows(hiop::hiopMatrix& A, long long M)
    {
        return A.m() == M ? 0 : 1;
    }

    global_ordinal_type matrixNumCols(hiop::hiopMatrix& A, long long N)
    {
        return A.n() == N ? 0 : 1;
    }

    int matrixSetToZero(hiop::hiopMatrix& A)
    {
        local_ordinal_type M = getNumLocRows(&A);
        local_ordinal_type N = getNumLocCols(&A);

        A.setToZero();

        for(local_ordinal_type i=0; i<M; ++i)
            for(local_ordinal_type j=0; j<N; ++j)
                if(getElement(&A,i,j) != 0)
                {
                    std::cerr << "Element (" << i << "," << j << ") not set to zero\n";
                    return 1;
                }

        return 0;
    }

protected:
    virtual void setElement(hiop::hiopMatrix* a, local_ordinal_type i, local_ordinal_type j, real_type val) = 0;
    virtual real_type getElement(hiop::hiopMatrix* a, local_ordinal_type i, local_ordinal_type j) = 0;
    virtual local_ordinal_type getNumLocRows(hiop::hiopMatrix* a) = 0;
    virtual local_ordinal_type getNumLocCols(hiop::hiopMatrix* a) = 0;

};

} // namespace hiopTest
