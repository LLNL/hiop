#include <iostream>
#include <hiopMatrix.hpp>

#include "LinAlg/matrixTestsDense.hpp"

int main()
{
    long long M = 10;  // rows
    long long N = 100; // columns
    int fail = 0;

    // Test dense matrix
    {
        hiop::hiopMatrixDense A(M, N);
        hiop::tests::MatrixTestsDense test;

        fail += test.matrixNumRows(A, M);
        fail += test.matrixNumCols(A, N);
    }

    // Test RAJA matrix
    {
        // Code here ...
    }




    if(fail)
        std::cout << "Matrix tests failed\n";
    else
        std::cout << "Matrix tests passed\n";

    return fail;
}
