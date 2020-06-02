// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// Written by Cosmin G. Petra, petra1@llnl.gov.
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp 
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause). 
// Please also read “Additional BSD Notice” below.
//
// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:
// i. Redistributions of source code must retain the above copyright notice, this list 
// of conditions and the disclaimer below.
// ii. Redistributions in binary form must reproduce the above copyright notice, 
// this list of conditions and the disclaimer (as noted below) in the documentation and/or 
// other materials provided with the distribution.
// iii. Neither the name of the LLNS/LLNL nor the names of its contributors may be used to 
// endorse or promote products derived from this software without specific prior written 
// permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
// SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR 
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS 
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED 
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Additional BSD Notice
// 1. This notice is required to be provided under our contract with the U.S. Department 
// of Energy (DOE). This work was produced at Lawrence Livermore National Laboratory under 
// Contract No. DE-AC52-07NA27344 with the DOE.
// 2. Neither the United States Government nor Lawrence Livermore National Security, LLC 
// nor any of their employees, makes any warranty, express or implied, or assumes any 
// liability or responsibility for the accuracy, completeness, or usefulness of any 
// information, apparatus, product, or process disclosed, or represents that its use would
// not infringe privately-owned rights.
// 3. Also, reference herein to any specific commercial products, process, or services by 
// trade name, trademark, manufacturer or otherwise does not necessarily constitute or 
// imply its endorsement, recommendation, or favoring by the United States Government or 
// Lawrence Livermore National Security, LLC. The views and opinions of authors expressed 
// herein do not necessarily state or reflect those of the United States Government or 
// Lawrence Livermore National Security, LLC, and shall not be used for advertising or 
// product endorsement purposes.

/**
 * @file matrixTestsDense.cpp
 * 
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>,  PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 * 
 */

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

void MatrixTestsDense::setLocalElement(
        hiop::hiopVector* _x,
        const local_ordinal_type i,
        const real_type val)
{
    auto x = dynamic_cast<hiop::hiopVectorPar*>(_x);
    real_type* data = x->local_data();
    data[i] = val;
}

/// Method to set a single row of matrix to a constant value
void MatrixTestsDense::setLocalRow(
        hiop::hiopMatrixDense* A,
        const local_ordinal_type row,
        const real_type val)
{
    const local_ordinal_type N = getNumLocCols(A);
    for (int i=0; i<N; i++)
    {
        setLocalElement(A, row, i, val);
    }
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
    (void) A;
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
    (void) A; // surpresses waring as A is not needed here without MPI
    local_row = row;
    local_col = col;
    return true;
#endif
}

// End helper methods

} // namespace hiop::tests
