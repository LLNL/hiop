#include <iostream>
#include <hiopVector.hpp>

#include "LinAlg/vectorTestsPar.hpp"

int main()
{
    int N = 1000;
    int fail = 0;

    hiop::hiopVector* x = new hiop::hiopVectorPar(N);
    hiopTest::VectorTestsPar* test = new hiopTest::VectorTestsPar();

    fail += test->testGetSize(*x, N);
    fail += test->testSetToConstant(*x);

    delete x;

    if(fail)
        std::cout << "Tests failed\n";
    else
        std::cout << "Tests passed\n";

    return fail;
}
