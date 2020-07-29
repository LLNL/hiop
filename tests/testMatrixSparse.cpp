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
 *
 */
#include <iostream>
#include <cassert>
#include <cstring>

#include <hiopVector.hpp>
#include <hiopMatrixDenseRowMajor.hpp>
#include "LinAlg/matrixTestsSparseTriplet.hpp"

using namespace hiop::tests;

/// Generate a sparse matrix in triplet format
void initializeSparseTriplet(hiop::hiopMatrixSparseTriplet &A, local_ordinal_type entries_per_row)
{
  local_ordinal_type * iRow = A.i_row();
  local_ordinal_type * jCol = A.j_col();
  double * val = A.M();

  local_ordinal_type m = A.m();
  local_ordinal_type n = A.n();

  assert(A.numberOfNonzeros() == m * entries_per_row && "Matrix initialized with insufficent number of non-zero entries");

  for(local_ordinal_type row = 0, col = 0, i = 0; row < m; row++, col = 0) 
  {
    for(local_ordinal_type j=0; j<entries_per_row-1; i++, j++, col += n / entries_per_row)
    {
      iRow[i] = row;
      jCol[i] = col;
      val[i] = one;
    }

    iRow[i] = row;
    jCol[i] = n-1;
    val[i++] = one;
    
  }
}

int main(int argc, char** argv)
{
  if(argc > 1)
    std::cout << "Executable " << argv[0] << " doesn't take any input.";

  local_ordinal_type M_local = 5;
  local_ordinal_type N_local = 10*M_local;

  // Sparse matrix is not distributed
  global_ordinal_type M_global = M_local;
  global_ordinal_type N_global = N_local;

  int fail = 0;

  // Test Sparse Triplet Matrix
  {
    std::cout << "Testing hiopMatrixSparseTriplet" << "\n";
    hiop::tests::MatrixTestsSparseTriplet test;

    // Establishing sparsity pattern and initializing Matrix
    local_ordinal_type entries_per_row = 5;
    local_ordinal_type nnz = M_local * entries_per_row;

    hiop::hiopMatrixSparseTriplet mxn_sparse(M_local, N_local, nnz);

    initializeSparseTriplet(mxn_sparse, entries_per_row);
  
    hiop::hiopVectorPar vec_m(M_global);
    hiop::hiopVectorPar vec_n(N_global);

    /// @see LinAlg/matrixTestsSparseTriplet.hpp for reasons why some tests are implemented/not implemented
    fail += test.matrixNumRows(mxn_sparse, M_global);
    fail += test.matrixNumCols(mxn_sparse, N_global);
    fail += test.matrixSetToZero(mxn_sparse);
    fail += test.matrixSetToConstant(mxn_sparse);
    fail += test.matrixTimesVec(mxn_sparse, vec_m, vec_n);
    fail += test.matrixTransTimesVec(mxn_sparse, vec_m, vec_n);
    fail += test.matrixMaxAbsValue(mxn_sparse);
    fail += test.matrixIsFinite(mxn_sparse);

    // Need a dense matrix to store the output of the following tests
    global_ordinal_type W_delta = M_global * 5;
    hiop::hiopMatrixDenseRowMajor W_dense(M_global + W_delta, N_global + W_delta);

    // local_ordinal_type test_offset = 10;
    local_ordinal_type test_offset = 4;
    fail += test.tripletAddMDinvMtransToDiagBlockOfSymDeMatUTri(mxn_sparse, vec_n, W_dense, test_offset);

    // Initialise other sparse Matrix
    // TODO: Since this is specific for the next test only, perhaps move it within the test.
    local_ordinal_type M2 = M_global * 2;
    nnz = M2 * (entries_per_row);

    hiop::hiopMatrixSparseTriplet m2xn_sparse(M2, N_global, nnz);
    initializeSparseTriplet(m2xn_sparse, entries_per_row);

    // Set offsets where to insert sparse matrix
    local_ordinal_type i_offset = 1;
    local_ordinal_type j_offset = M2 + 1;

    fail += test.tripletAddMDinvNtransToSymDeMatUTri(mxn_sparse, m2xn_sparse, vec_n, W_dense, i_offset, j_offset);
  }

  // Test RAJA matrix
  {
    // Code here ...
  }

  if(fail)
  {
    std::cout << fail << " matrix tests failed\n";
  }
  else
  {
    std::cout << "All matrix tests passed\n";
  }

  return fail;
}
