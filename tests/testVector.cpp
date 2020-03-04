#include <iostream>
#include <assert.h>

#ifdef HIOP_USE_MPI
#include <mpi.h>
#endif

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
    int rank = 0;
    int numRanks = 1;
    MPI_Comm comm = MPI_COMM_NULL;

#ifdef HIOP_USE_MPI
    int err;
    comm = MPI_COMM_WORLD;
    err = MPI_Init(&argc, &argv);           assert(MPI_SUCCESS == err);
    err = MPI_Comm_rank(comm,&rank);        assert(MPI_SUCCESS == err);
    err = MPI_Comm_size(comm,&numRanks);    assert(MPI_SUCCESS == err);

    if (0 == rank)
        printf("Support for MPI is enabled\n");
#endif

    int N = 1000;
    int fail = 0;

    // Test parallel vector
    {
        hiop::hiopVectorPar x(N, nullptr, comm);
        auto y = x.alloc_clone();
        auto z = x.alloc_clone();

        hiop::tests::VectorTestsPar test;

        fail += test.vectorGetSize(x, N);
        fail += test.vectorSetToConstant(x);
        fail += test.vectorSetToZero(x);
        fail += test.vectorScale(x);

        fail += test.vectorSelectPattern(x, *y);
        fail += test.vectorCopyTo(x, *y);
        fail += test.vectorCopyFrom(x, *y);
        fail += test.vectorComponentDiv(x, *y);
        fail += test.vectorComponentMult(x, *y);
        fail += test.vectorComponentDiv_p_selectPattern(x, *y, *z);
    }

    // Test RAJA vector
    {
        //         hiop::hiopVectorRAJA x(N);
        //         hiop::tests::VectorTestsRAJA test;
        //
        //         fail += test.testGetSize(x, N);
        //         fail += test.testSetToConstant(x);
    }

    if (rank == 0)
    {
        if(fail)
            std::cout << "Tests failed\n";
        else
            std::cout << "Tests passed\n";
    }

#ifdef HIOP_USE_MPI
    MPI_Finalize();
#endif

    return fail;
}
