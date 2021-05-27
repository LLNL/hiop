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
 * @file testMatrixSparse.cpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>, PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 * @author Cameron Rutherford <robert.rutherford@pnnl.gov>, PNNL
 * @author Jake K. Ryan <jake.ryan@pnnl.gov>, PNNL
 *
 */
#include <iostream>
#include <cassert>
#include <cstring>

#include <hiopOptions.hpp>
#include <hiopLinAlgFactory.hpp>
#include <hiopVectorPar.hpp>
#include <hiopMatrixDenseRowMajor.hpp>

#include "LinAlg/matrixTestsSymSparseTriplet.hpp"

#ifdef HIOP_USE_RAJA
#include <hiopVectorRajaPar.hpp>
#include <hiopMatrixRajaDense.hpp>
#include "LinAlg/matrixTestsRajaSymSparseTriplet.hpp"
#endif

using namespace hiop::tests;

/**
 * @brief Initialize sparse matrix with a homogeneous pattern to test a
 * realistic use-case.
 */
void initializeSymSparseMat(hiop::hiopMatrixSparse* mat)
{
  auto* A = dynamic_cast<hiop::hiopMatrixSymSparseTriplet*>(mat);
  local_ordinal_type* iRow = A->i_row();
  local_ordinal_type* jCol = A->j_col();
  double* val = A->M();
  const auto nnz = A->numberOfNonzeros();
  int nonZerosUsed = 0;

  local_ordinal_type m = A->m();
  local_ordinal_type n = A->n();

  int num_entries = n * m - (m * (m - 1) / 2);
  int density = num_entries / nnz;

  auto iRow_idx = 0;
  auto jCol_idx = 0;

  for (auto i = 0; i < num_entries; i++)
  {
    if(i % density == 0)
    {
      iRow[nonZerosUsed] = iRow_idx;
      jCol[nonZerosUsed] = jCol_idx;
      val[nonZerosUsed] = i;
      nonZerosUsed++;
      if(nnz == nonZerosUsed)
      {
        break;
      } 
    }

    jCol_idx++;
    // If we are at the end of the current row
    if (jCol_idx % n == 0)
    {
      iRow_idx++;
      jCol_idx = iRow_idx;
    }
  }
  assert(nnz == nonZerosUsed && "incorrect amount of non-zeros in sparse sym matrix");
}

#ifdef HIOP_USE_RAJA
/**
 * @brief Initialize RAJA sparse matrix with a homogeneous pattern to test a
 * realistic use-case.
 */
void initializeRajaSymSparseMat(hiop::hiopMatrixSparse* mat)
{
  auto* A = dynamic_cast<hiop::hiopMatrixRajaSymSparseTriplet*>(mat);
  local_ordinal_type* iRow = A->i_row_host();
  local_ordinal_type* jCol = A->j_col_host();
  double* val = A->M_host();
  const auto nnz = A->numberOfNonzeros();
  int nonZerosUsed = 0;

  local_ordinal_type m = A->m();
  local_ordinal_type n = A->n();

  int num_entries = n * m - (m * (m - 1) / 2);
  int density = num_entries / nnz;

  auto iRow_idx = 0;
  auto jCol_idx = 0;

  for (auto i = 0; i < num_entries; i++)
  {
    if (i % density == 0)
    {
      iRow[nonZerosUsed] = iRow_idx;
      jCol[nonZerosUsed] = jCol_idx;
      val[nonZerosUsed] = i;
      nonZerosUsed++;
      if (nonZerosUsed == nnz)
        break;
    }

    jCol_idx++;
    // If we are at the end of the current row
    if (jCol_idx % n == 0)
    {
      iRow_idx++;
      jCol_idx = iRow_idx;
    }
  }
  assert(nnz == nonZerosUsed && "incorrect amount of non-zeros in sparse sym matrix");
  A->copyToDev();
}
#endif

int main(int argc, char** argv)
{
  if(argc > 1)
    std::cout << "Executable " << argv[0] << " doesn't take any input.";

  hiop::hiopOptions options;

  local_ordinal_type M_local = 50;

  // Sparse matrix is not distributed
  global_ordinal_type M_global = M_local;

  int fail = 0;

  // Test sparse matrix
  {
    std::cout << "\nTesting hiopMatrixSymSparseTriplet\n";
    hiop::tests::MatrixTestsSymSparseTriplet test;

    // Establishing sparsity pattern and initializing Matrix
    local_ordinal_type entries_per_row = 5;
    local_ordinal_type nnz = M_global * entries_per_row;

    hiop::hiopVectorPar vec_m(M_global);
    hiop::hiopVectorPar vec_m_2(M_global);
    hiop::hiopMatrixDenseRowMajor mxm_dense(2 * M_global, 2 * M_global);

    hiop::hiopMatrixSparse* m_sym = 
      hiop::LinearAlgebraFactory::createMatrixSymSparse(M_global, nnz);
    initializeSymSparseMat(m_sym);

    local_ordinal_type nnz_m2 = m_sym->numberOfOffDiagNonzeros() + M_global;
    hiop::hiopMatrixSparse* m2_sym = 
      hiop::LinearAlgebraFactory::createMatrixSymSparse(2*M_global, nnz_m2);

    fail += test.matrixTimesVec(*m_sym, vec_m, vec_m_2);
    fail += test.matrixAddUpperTriangleToSymDenseMatrixUpperTriangle(mxm_dense, *m_sym);
    fail += test.matrixStartingAtAddSubDiagonalToStartingAt(vec_m, *m_sym);

    fail += test.matrix_set_Hess_FR(mxm_dense, *m2_sym, *m_sym, vec_m);

    // Destroy testing objects
    delete m_sym;
    delete m2_sym;
  }

#ifdef HIOP_USE_RAJA
  // Test RAJA sparse matrix
  {
    std::cout << "\nTesting hiopMatrixRajaSymSparseTriplet\n";

    options.SetStringValue("mem_space", "device");
    hiop::LinearAlgebraFactory::set_mem_space(options.GetString("mem_space"));
    std::string mem_space = hiop::LinearAlgebraFactory::get_mem_space();

    hiop::tests::MatrixTestsRajaSymSparseTriplet test;
    
    // Establishing sparsity pattern and initializing Matrix
    local_ordinal_type entries_per_row = 5;
    local_ordinal_type nnz = M_local * entries_per_row;

    hiop::hiopVectorRajaPar vec_m(M_global, mem_space);
    hiop::hiopVectorRajaPar vec_m_2(M_global, mem_space);
    hiop::hiopMatrixRajaDense mxm_dense(2 * M_global, 2 * M_global, mem_space);

    hiop::hiopMatrixSparse* m_sym = 
      hiop::LinearAlgebraFactory::createMatrixSymSparse(M_local, nnz);
    initializeRajaSymSparseMat(m_sym);

    local_ordinal_type nnz_m2 = m_sym->numberOfOffDiagNonzeros() + M_global;
    hiop::hiopMatrixSparse* m2_sym = 
      hiop::LinearAlgebraFactory::createMatrixSymSparse(2*M_global, nnz_m2);

    fail += test.matrixTimesVec(*m_sym, vec_m, vec_m_2);
    fail += test.matrixAddUpperTriangleToSymDenseMatrixUpperTriangle(mxm_dense, *m_sym);
    fail += test.matrixStartingAtAddSubDiagonalToStartingAt(vec_m, *m_sym);

    fail += test.matrix_set_Hess_FR(mxm_dense, *m2_sym, *m_sym, vec_m);

    // Destroy testing objects
    delete m_sym;
    delete m2_sym;

    // Set memory space back to default value
    options.SetStringValue("mem_space", "default");
  }
#endif

  if(fail)
  {
    std::cout << "\n" << fail << " sparse matrix tests failed!\n\n";
  }
  else
  {
    std::cout << "\nAll sparse matrix tests passed!\n\n";
  }

  return fail;
}
