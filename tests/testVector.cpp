#include <iostream>
#include <hiopVector.hpp>

#include "LinAlg/vectorTestsPar.hpp"
// #include "LinAlg/vectorTestsRAJA.hpp"

int main()
{
    int N = 1000;
    int fail = 0;

    // Test parallel vector
    {
        hiop::hiopVector* x = new hiop::hiopVectorPar(N);
        hiopTest::VectorTestsPar test;

        fail += test.testGetSize(*x, N);
        fail += test.testSetToConstant(*x);

        delete x;
    }

    // Test RAJA vector
    {
        //         hiop::hiopVector* x = new hiop::hiopVectorRAJA(N);
        //         hiopTest::VectorTestsRAJA test;
        //
        //         fail += test.testGetSize(*x, N);
        //         fail += test.testSetToConstant(*x);
        //
        //         delete x;
    }




    if(fail)
        std::cout << "Tests failed\n";
    else
        std::cout << "Tests passed\n";

    return fail;
}
