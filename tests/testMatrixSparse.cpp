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
#include <hiopVector.hpp>
#include <hiopMatrixDense.hpp>
#include "LinAlg/matrixTestsSparseTriplet.hpp"

#ifdef HIOP_USE_RAJA
#include "LinAlg/matrixTestsRajaSparseTriplet.hpp"
#endif

template <typename T>
static int runTests(const char* mem_space);

int main(int argc, char** argv)
{
  using namespace hiop::tests;
  using hiop::tests::global_ordinal_type;
  
  if(argc > 1)
    std::cout << "Executable " << argv[0] << " doesn't take any input.";

  hiop::hiopOptions options;

  int fail = 0;

  //
  // Test HiOp Sparse Matrices
  //
  std::cout << "\nTesting HiOp default sparse matrix implementation:\n";
  fail += runTests<MatrixTestsSparseTriplet>("default");
#ifdef HIOP_USE_RAJA
#ifdef HIOP_USE_GPU
  std::cout << "\nTesting HiOp RAJA sparse matrix implementation:\n";
  std::cout << "  ... using device memory space:\n";
  fail += runTests<MatrixTestsRajaSparseTriplet>("device");
  std::cout << "\nTesting HiOp RAJA sparse matrix implementation:\n";
  std::cout << "  ... using unified virtual memory space:\n";
  fail += runTests<MatrixTestsRajaSparseTriplet>("um");
#else
  std::cout << "\nTesting HiOp RAJA sparse matrix implementation:\n";
  std::cout << "  ... unified host memory space:\n";
  fail += runTests<MatrixTestsRajaSparseTriplet>("host");
#endif // GPU
#endif // RAJA

  if(fail)
  {
    std::cout << "\n" << fail << " sparse matrix tests failed\n\n";
  }
  else
  {
    std::cout << "\nAll sparse matrix tests passed\n\n";
  }

  return fail;
}

/// Driver for all sparse matrix tests
template <typename T>
static int runTests(const char* mem_space)
{
  using namespace hiop;
  using hiop::tests::local_ordinal_type;
  using hiop::tests::global_ordinal_type;

  T test;

  hiopOptions options;
  options.SetStringValue("mem_space", mem_space);
  LinearAlgebraFactory::set_mem_space(mem_space);

  int fail = 0;

  local_ordinal_type M_local = 5;
  local_ordinal_type N_local = 10 * M_local;

  // Establishing sparsity pattern and initializing Matrix
  local_ordinal_type entries_per_row = 5;
  local_ordinal_type nnz = M_local * entries_per_row;

  // Sparse matrix is not distributed
  global_ordinal_type M_global = M_local;
  global_ordinal_type N_global = N_local;

  hiopMatrixSparse* mxn_sparse = LinearAlgebraFactory::createMatrixSparse(M_local, N_local, nnz);
  test.initializeMatrix(mxn_sparse, entries_per_row);

  hiopVector* vec_m = LinearAlgebraFactory::createVector(M_global);
  hiopVector* vec_n = LinearAlgebraFactory::createVector(N_global);

  /// @see LinAlg/matrixTestsSparseTriplet.hpp for reasons why some tests are implemented/not implemented
  fail += test.matrixNumRows(*mxn_sparse, M_global);
  fail += test.matrixNumCols(*mxn_sparse, N_global);
  fail += test.matrixSetToZero(*mxn_sparse);
  fail += test.matrixSetToConstant(*mxn_sparse);
  fail += test.matrixMaxAbsValue(*mxn_sparse);
  fail += test.matrixIsFinite(*mxn_sparse);
  fail += test.matrixTimesVec(*mxn_sparse, *vec_m, *vec_n);
  fail += test.matrixTransTimesVec(*mxn_sparse, *vec_m, *vec_n);

  // Need a dense matrix to store the output of the following tests
  global_ordinal_type W_delta = M_global * 10;
  hiopMatrixDense* W_dense = LinearAlgebraFactory::createMatrixDense(N_global + W_delta, N_global + W_delta);

  local_ordinal_type test_offset = 4;
  fail += test.matrixAddMDinvMtransToDiagBlockOfSymDeMatUTri(*mxn_sparse, *vec_n, *W_dense, test_offset);

  // testing adding sparse matrix to the upper triangular area of a symmetric dense matrix    
  fail += test.transAddToSymDenseMatrixUpperTriangle(*W_dense, *mxn_sparse);

  // Initialise another sparse Matrix
  local_ordinal_type M2 = M_global * 2;
  nnz = M2 * (entries_per_row);

  hiopMatrixSparse* m2xn_sparse = LinearAlgebraFactory::createMatrixSparse(M2, N_global, nnz);
  test.initializeMatrix(m2xn_sparse, entries_per_row);

  // Set offsets where to insert sparse matrix
  local_ordinal_type i_offset = 1;
  local_ordinal_type j_offset = M2 + 1;

  fail += test.matrixAddMDinvNtransToSymDeMatUTri(*mxn_sparse, *m2xn_sparse, *vec_n, *W_dense, i_offset, j_offset);

  //
  // Perform hipoMatrixSparseTriplet specific tests
  //
  //auto * test_triplet = dynamic_cast<tests::MatrixTestsSparseTriplet *>(&test);
  if (dynamic_cast<tests::MatrixTestsSparseTriplet *>(&test))
  {
    fail += test.copyRowsBlockFrom(*mxn_sparse, *m2xn_sparse,0, 1, M_global-1, mxn_sparse->numberOfNonzeros()-entries_per_row);
  }

  delete mxn_sparse;
  delete m2xn_sparse;
  delete W_dense;
  delete vec_m;
  delete vec_n;

  return fail;
}