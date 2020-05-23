#include <hiopMatrix.hpp>
#include "matrixTestsDense.hpp"

namespace hiop::tests {

/// Method to set matrix _A_ element (i,j) to _val_.
/// First need to retrieve hiopMatrixDense from the abstract interface
void MatrixTestsDense::setLocalElement(
        hiop::hiopMatrix* A,
        local_ordinal_type i,
        local_ordinal_type j,
        real_type val)
{
    hiop::hiopMatrixDense* amat = dynamic_cast<hiop::hiopMatrixDense*>(A);
    real_type** data = amat->get_M();
    data[i][j] = val;
}

/// Returns element (i,j) of matrix _A_.
/// First need to retrieve hiopMatrixDense from the abstract interface
real_type MatrixTestsDense::getLocalElement(
        const hiop::hiopMatrix* A,
        local_ordinal_type i,
        local_ordinal_type j)
{
    const hiop::hiopMatrixDense* amat = dynamic_cast<const hiop::hiopMatrixDense*>(A);
    return amat->local_data()[i][j];
}

/// Returns element _i_ of vector _x_.
/// First need to retrieve hiopVectorPar from the abstract interface
real_type MatrixTestsDense::getLocalElement(
        const hiop::hiopVector* x,
        local_ordinal_type i)
{
    const hiop::hiopVectorPar* xvec = dynamic_cast<const hiop::hiopVectorPar*>(x);
    return xvec->local_data_const()[i];
}

local_ordinal_type MatrixTestsDense::getNumLocRows(hiop::hiopMatrix* A)
{
    hiop::hiopMatrixDense* amat = dynamic_cast<hiop::hiopMatrixDense*>(A);
    return amat->get_local_size_m();
    //                         ^^^
}

local_ordinal_type MatrixTestsDense::getNumLocCols(hiop::hiopMatrix* A)
{
    hiop::hiopMatrixDense* amat = dynamic_cast<hiop::hiopMatrixDense*>(A);
    return amat->get_local_size_n();
    //                         ^^^
}

/// Returns size of local data array for vector _x_
int MatrixTestsDense::getLocalSize(const hiop::hiopVector* x)
{
    const hiop::hiopVectorPar* xvec = dynamic_cast<const hiop::hiopVectorPar*>(x);
    return static_cast<int>(xvec->get_local_size());
}

#ifdef HIOP_USE_MPI
/// Get communicator
MPI_Comm MatrixTestsDense::getMPIComm(hiop::hiopMatrix* _A)
{
    const hiop::hiopMatrixDense* A = dynamic_cast<const hiop::hiopMatrixDense*>(_A);
    return A->get_mpi_comm();
}
#endif

// Every rank returns failure if any individual rank fails
bool MatrixTestsDense::reduceReturn(int failures, hiop::hiopMatrix* A)
{
    int fail = 0;

#ifdef HIOP_USE_MPI
    MPI_Allreduce(&failures, &fail, 1, MPI_INT, MPI_SUM, getMPIComm(A));
#else
    fail = failures;
#endif

    return (fail != 0);
}

[[nodiscard]]
int MatrixTestsDense::verifyAnswer(hiop::hiopMatrix* A, const double answer)
{
    const local_ordinal_type M = getNumLocRows(A);
    const local_ordinal_type N = getNumLocCols(A);
    int fail = 0;
    for (local_ordinal_type i=0; i<M; i++)
    {
        for (local_ordinal_type j=0; j<N; j++)
        {
            if (!isEqual(getLocalElement(A, i, j), answer))
            {
                std::cout << i << " " << j << " = " << getLocalElement(A, i, j) << " != " << answer << "\n";
                fail++;
            }
        }
    }
    return fail;
}

/*
 * Pass a function-like object to calculate the expected
 * answer dynamically, based on the row and column
 */
[[nodiscard]]
int MatrixTestsDense::verifyAnswer(
            hiop::hiopMatrix* A,
            std::function<real_type(local_ordinal_type, local_ordinal_type)> expect)
{
    const local_ordinal_type M = getNumLocRows(A);
    const local_ordinal_type N = getNumLocCols(A);
    int fail = 0;
    for (local_ordinal_type i=0; i<M; i++)
    {
        for (local_ordinal_type j=0; j<N; j++)
        {
            if (!isEqual(getLocalElement(A, i, j), expect(i, j)))
            {
                fail++;
            }
        }
    }

    return fail;
}

/// Checks if _local_ vector elements are set to `answer`.
[[nodiscard]]
int MatrixTestsDense::verifyAnswer(hiop::hiopVector* x, double answer)
{
    const local_ordinal_type N = getLocalSize(x);

    int local_fail = 0;
    for(local_ordinal_type i=0; i<N; ++i)
    {
        if(!isEqual(getLocalElement(x, i), answer))
        {
            ++local_fail;
        }
    }

    return local_fail;
}

[[nodiscard]]
int MatrixTestsDense::verifyAnswer(
    hiop::hiopVector* x,
    std::function<real_type(local_ordinal_type)> expect)
{
    const local_ordinal_type N = getLocalSize(x);

    int local_fail = 0;
    for (int i=0; i<N; i++)
    {
        if(!isEqual(getLocalElement(x, i), expect(i)))
        {
            ++local_fail;
        }
    }
    return local_fail;
}

bool MatrixTestsDense::globalToLocalMap(
    hiop::hiopMatrix* A,
    const global_ordinal_type row,
    const global_ordinal_type col,
    local_ordinal_type& local_row,
    local_ordinal_type& local_col)
{
#ifdef HIOP_USE_MPI
    int rank = 0;
    MPI_Comm comm = getMPIComm(A);
    MPI_Comm_rank(comm, &rank);
    const local_ordinal_type n_local = getNumLocCols(A);
    const global_ordinal_type local_col_start = n_local * rank;
    if (col >= local_col_start && col < local_col_start + n_local)
    {
        local_row = row;
        local_col = col % n_local;
        return true;
    }
    else
    {
        return false;
    }
#else
    local_row = row;
    local_col = col;
    return true;
#endif
}

} // namespace hiop::tests
