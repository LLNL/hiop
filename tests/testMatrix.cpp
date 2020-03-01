#include <iostream>
#include <hiopMatrix.hpp>

#include "LinAlg/matrixTestsPar.hpp"

int main()
{
    int N = 100;
    int fail = 0;

    // Test dense matrix
    {
        hiop::hiopMatrix* m = new hiop::hiopMatrixDense(N, N);
        hiopTest::MatrixTestsDense test;

        delete x;
    }

    // Test RAJA matrix
    {
        // Code here ...
    }




    if(fail)
        std::cout << "Tests failed\n";
    else
        std::cout << "Tests passed\n";

    return fail;
}
