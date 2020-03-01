#include <iostream>
#include <hiopVector.hpp>

#include "LinAlg/vectorTestsPar.hpp"
// #include "LinAlg/vectorTestsRAJA.hpp"

/**
 * @brief Main body of vector implementation testing code.
 *
 * @todo The code should support MPI tests when appropriate input is passed
 * on the command line.
 *
 */
int main(int argc, char** argv)
{
    int N = 1000;
    int fail = 0;

    // Test parallel vector
    {
        hiop::hiopVectorPar x(N);
        hiop::tests::VectorTestsPar test;

        fail += test.vectorGetSize(x, N);
        fail += test.vectorSetToConstant(x);
    }

    // Test RAJA vector
    {
        //         hiop::hiopVectorRAJA x(N);
        //         hiop::tests::VectorTestsRAJA test;
        //
        //         fail += test.testGetSize(x, N);
        //         fail += test.testSetToConstant(x);
    }

    if(fail)
        std::cout << "Tests failed\n";
    else
        std::cout << "Tests passed\n";

    return fail;
}
