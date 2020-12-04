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
 * @file MatrixTestsRajaDense.cpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>,  PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 *
 */

#include <hiopMatrixRajaDense.hpp>
#include <hiopVectorRajaPar.hpp>
#include "matrixTestsRajaDense.hpp"

namespace hiop { namespace tests {

/// Method to set matrix _A_ element (i,j) to _val_.
/// First need to retrieve hiopMatrixDense from the abstract interface
void MatrixTestsRajaDense::setLocalElement(
    hiop::hiopMatrixDense* A,
    local_ordinal_type i,
    local_ordinal_type j,
    real_type val)
{
  hiop::hiopMatrixRajaDense* amat = dynamic_cast<hiop::hiopMatrixRajaDense*>(A);
  if(amat != nullptr)
  {
    amat->copyFromDev();
    real_type* data = amat->local_data_host(); 
    data[i*amat->get_local_size_n() + j] = val;
    amat->copyToDev();
  }
  else THROW_NULL_DEREF;
}

void MatrixTestsRajaDense::setLocalElement(
    hiop::hiopVector* xvec,
    const local_ordinal_type i,
    const real_type val)
{
  auto* x = dynamic_cast<hiop::hiopVectorRajaPar*>(xvec);
  if(x != nullptr)
  {
    x->copyFromDev();
    real_type* data = x->local_data_host();
    data[i] = val;
    x->copyToDev();
  }
  else THROW_NULL_DEREF;
}

/// Method to set a single row of matrix to a constant value
void MatrixTestsRajaDense::setLocalRow(
    hiop::hiopMatrixDense* Amat,
    const local_ordinal_type row,
    const real_type val)
{
  hiop::hiopMatrixRajaDense* A = dynamic_cast<hiop::hiopMatrixRajaDense*>(Amat);
  const local_ordinal_type N = getNumLocCols(A);
  A->copyFromDev();
  real_type* local_data = A->local_data_host();
  for (int j=0; j<N; j++)
  {
    local_data[row*N+j] =  val;
  }
  A->copyToDev();
}

/// Returns element (i,j) of matrix _A_.
/// First need to retrieve hiopMatrixDense from the abstract interface
real_type MatrixTestsRajaDense::getLocalElement(
    const hiop::hiopMatrixDense* A,
    local_ordinal_type i,
    local_ordinal_type j)
{
  const hiop::hiopMatrixRajaDense* am = dynamic_cast<const hiop::hiopMatrixRajaDense*>(A);
  hiop::hiopMatrixRajaDense* amat = const_cast<hiop::hiopMatrixRajaDense*>(am);
  if(amat != nullptr)
  {
    amat->copyFromDev();
    return amat->local_data_host()[i*amat->get_local_size_n() + j];
  }
  else THROW_NULL_DEREF;
}

/// Returns element _i_ of vector _x_.
/// First need to retrieve hiopVectorPar from the abstract interface
real_type MatrixTestsRajaDense::getLocalElement(
    const hiop::hiopVector* x,
    local_ordinal_type i)
{
  const hiop::hiopVectorRajaPar* xvec_const = dynamic_cast<const hiop::hiopVectorRajaPar*>(x);
  hiop::hiopVectorRajaPar* xvec = const_cast<hiop::hiopVectorRajaPar*>(xvec_const);
  if(xvec != nullptr)
  {
    xvec->copyFromDev();
    return xvec->local_data_host_const()[i];
  }
  else THROW_NULL_DEREF;
}

local_ordinal_type MatrixTestsRajaDense::getNumLocRows(hiop::hiopMatrixDense* A)
{
  hiop::hiopMatrixDense* amat = dynamic_cast<hiop::hiopMatrixDense*>(A);
  if(amat != nullptr)
    return amat->get_local_size_m();
    //                         ^^^
  else THROW_NULL_DEREF;
}

local_ordinal_type MatrixTestsRajaDense::getNumLocCols(hiop::hiopMatrixDense* A)
{
  hiop::hiopMatrixDense* amat = dynamic_cast<hiop::hiopMatrixDense*>(A);
  if(amat != nullptr)
    return amat->get_local_size_n();
    //                         ^^^
  else THROW_NULL_DEREF;
}

/// Returns size of local data array for vector _x_
int MatrixTestsRajaDense::getLocalSize(const hiop::hiopVector* x)
{
  const hiop::hiopVectorRajaPar* xvec = dynamic_cast<const hiop::hiopVectorRajaPar*>(x);
  if(xvec != nullptr)
    return static_cast<int>(xvec->get_local_size());
  else THROW_NULL_DEREF;
}

#ifdef HIOP_USE_MPI
/// Get communicator
MPI_Comm MatrixTestsRajaDense::getMPIComm(hiop::hiopMatrixDense* _A)
{
  const hiop::hiopMatrixDense* A = dynamic_cast<const hiop::hiopMatrixDense*>(_A);
  if(A != nullptr)
    return A->get_mpi_comm();
  else THROW_NULL_DEREF;
}
#endif

/// Returns pointer to local data block of matrix _A_.
/// First need to retrieve hiopMatrixRajaDense from the abstract interface
real_type* MatrixTestsRajaDense::getLocalData(hiop::hiopMatrixDense* A)
{
  hiop::hiopMatrixRajaDense* amat = dynamic_cast<hiop::hiopMatrixRajaDense*>(A);
  if(amat != nullptr)
  {
    return amat->local_data();
  }
  else THROW_NULL_DEREF;
}

// Every rank returns failure if any individual rank fails
bool MatrixTestsRajaDense::reduceReturn(int failures, hiop::hiopMatrixDense* A)
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
int MatrixTestsRajaDense::verifyAnswer(hiop::hiopMatrixDense* Amat, const double answer)
{
  hiop::hiopMatrixRajaDense* A = dynamic_cast<hiop::hiopMatrixRajaDense*>(Amat);
  const local_ordinal_type M = getNumLocRows(A);
  const local_ordinal_type N = getNumLocCols(A);

  // Copy data to the host mirror
  A->copyFromDev();
  // Get array of pointers to dense matrix rows
  double* local_matrix_data = A->local_data_host();

  int fail = 0;
  for (local_ordinal_type i=0; i<M; i++)
  {
    for (local_ordinal_type j=0; j<N; j++)
    {
      if (!isEqual(local_matrix_data[i*N+j], answer))
      {
        std::cout << i << " " << j << " = " << local_matrix_data[i*N+j] << " != " << answer << "\n";
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
int MatrixTestsRajaDense::verifyAnswer(
    hiop::hiopMatrixDense* Amat,
    std::function<real_type(local_ordinal_type, local_ordinal_type)> expect)
{
  // const local_ordinal_type M = getNumLocRows(A);
  // const local_ordinal_type N = getNumLocCols(A);
  hiop::hiopMatrixRajaDense* A = dynamic_cast<hiop::hiopMatrixRajaDense*>(Amat);
  const local_ordinal_type M = getNumLocRows(A);
  const local_ordinal_type N = getNumLocCols(A);

  // Copy data to the host mirror
  A->copyFromDev();
  // Get array of pointers to dense matrix rows
  double* local_matrix_data = A->local_data_host();
  
  int fail = 0;
  for (local_ordinal_type i=0; i<M; i++)
  {
    for (local_ordinal_type j=0; j<N; j++)
    {
      if (!isEqual(local_matrix_data[i*N+j], expect(i, j)))
      {
        std::cout << i << ", " << j << ": = " << local_matrix_data[i*N+j] << " != " << expect(i, j) << "\n";
        fail++;
      }
    }
  }
  return fail;
}

/// Checks if _local_ vector elements are set to `answer`.
  [[nodiscard]]
int MatrixTestsRajaDense::verifyAnswer(hiop::hiopVector* xvec, double answer)
{
  auto* x = dynamic_cast<hiop::hiopVectorRajaPar*>(xvec);
  const local_ordinal_type N = getLocalSize(x);

  // Copy vector local data to the host mirror
  x->copyFromDev();
  // Get raw pointer to the host mirror
  const real_type* local_data = x->local_data_host_const();

  int local_fail = 0;
  for(local_ordinal_type i=0; i<N; ++i)
  {
    if(!isEqual(local_data[i], answer))
    {
      ++local_fail;
    }
  }

  return local_fail;
}

  [[nodiscard]]
int MatrixTestsRajaDense::verifyAnswer(
    hiop::hiopVector* xvec,
    std::function<real_type(local_ordinal_type)> expect)
{
  auto* x = dynamic_cast<hiop::hiopVectorRajaPar*>(xvec);
  const local_ordinal_type N = getLocalSize(x);

  // Copy vector local data to the host mirror
  x->copyFromDev();
  // Get raw pointer to the host mirror
  const real_type* local_data = x->local_data_host_const();

  int local_fail = 0;
  for (int i=0; i<N; i++)
  {
    if(!isEqual(local_data[i], expect(i)))
    {
      ++local_fail;
    }
  }
  return local_fail;
}

bool MatrixTestsRajaDense::globalToLocalMap(
    hiop::hiopMatrixDense* A,
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

}} // namespace hiop::tests
