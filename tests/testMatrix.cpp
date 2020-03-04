#include <iostream>
// #include <mpi.h>

#include <hiopMatrix.hpp>
#include "LinAlg/matrixTestsDense.hpp"

int main()
{
    int rank=0, numRanks=1;
// #ifdef HIOP_USE_MPI
//     int err;
//     err = MPI_Init(&argc, &argv);                  assert(MPI_SUCCESS==err);
//     err = MPI_Comm_rank(MPI_COMM_WORLD,&rank);     assert(MPI_SUCCESS==err);
//     err = MPI_Comm_size(MPI_COMM_WORLD,&numRanks); assert(MPI_SUCCESS==err);
//     if(0 == rank)
//         printf("Support for MPI is enabled\n");
// #endif

    long long M = 10;  // rows
    long long N = 100; // columns
    int fail = 0;

    // Test dense matrix
    {
        hiop::hiopMatrixDense A(M, N);
        hiop::tests::MatrixTestsDense test;

        // Fill in dense matrix A with ones
        /// @warning m() in hiopMatrixDense() is shadowing m() in hiopMatrix!
        /// This is a temporary solution and needs to be rewritten!
        double** data = A.local_data();
        for(int i=0; i<A.m(); ++i)
            for(int j=0; j<A.n(); ++j)
                data[i][j] = 1.0;

        fail += test.matrixNumRows(A, M);
        fail += test.matrixNumCols(A, N);
        fail += test.matrixSetToZero(A);
    }

    // Test RAJA matrix
    {
        // Code here ...
    }

    if (rank == 0)
        if(fail)
            std::cout << "Matrix tests failed\n";
        else
            std::cout << "Matrix tests passed\n";

// #ifdef HIOP_USE_MPI
//     MPI_Finalize();
// #endif

    return fail;
}
