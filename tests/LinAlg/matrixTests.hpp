#pragma once

#include <iostream>
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

    int matrixSetToZero(hiop::hiopMatrix& A)
    {
        int M = getNumLocRows(&A);
        int N = getNumLocCols(&A);

        A.setToZero();

        for(int i=0; i<M; ++i)
            for(int j=0; j<N; ++j)
                if(getElement(&A,i,j) != 0)
                {
                    std::cerr << "Element (" << i << "," << j << ") not set to zero\n";
                    return 1;
                }

        return 0;
    }

protected:
    virtual void setElement(hiop::hiopMatrix* a, int i, int j, double val) = 0;
    virtual double getElement(hiop::hiopMatrix* a, int i, int j) = 0;
    virtual int getNumLocRows(hiop::hiopMatrix* a) = 0;
    virtual int getNumLocCols(hiop::hiopMatrix* a) = 0;

};

} // namespace hiopTest
