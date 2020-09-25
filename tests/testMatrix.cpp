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
 * @file testMatrix.cpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>, PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 *
 */
#include <iostream>
#include <cassert>

#include <hiopOptions.hpp>
#include <hiopMPI.hpp>
#include <hiopLinAlgFactory.hpp>
#include <hiopVectorPar.hpp>
#include <hiopMatrixDenseRowMajor.hpp>
#include "LinAlg/matrixTestsDense.hpp"

#ifdef HIOP_USE_RAJA
#include <hiopVectorRajaPar.hpp>
#include <hiopMatrixRajaDense.hpp>
#include "LinAlg/matrixTestsRajaDense.hpp"
#endif

int main(int argc, char** argv)
{
  using hiop::tests::global_ordinal_type;

  int rank = 0;
  int numRanks = 1;
  MPI_Comm comm = MPI_COMM_SELF;

#ifdef HIOP_USE_MPI
  int err;
  err = MPI_Init(&argc, &argv);        assert(MPI_SUCCESS==err);
  comm = MPI_COMM_WORLD;
  err = MPI_Comm_rank(comm,&rank);     assert(MPI_SUCCESS==err);
  err = MPI_Comm_size(comm,&numRanks); assert(MPI_SUCCESS==err);
  if(0 == rank && MPI_SUCCESS == err)
    printf("Support for MPI is enabled\n");
#endif
  if(rank == 0 && argc > 1)
    std::cout << "Executable " << argv[0] << " doesn't take any input.";

  hiop::hiopOptions options;

  global_ordinal_type M_local = 50;
  global_ordinal_type K_local = 2 * M_local;
  global_ordinal_type N_local = 10 * M_local;

  // all distribution occurs column-wise, so any length
  // that will be used as a column of a matrix will have
  // to be scaled up by numRanks
  global_ordinal_type M_global = M_local * numRanks;
  global_ordinal_type K_global = K_local * numRanks;
  global_ordinal_type N_global = N_local * numRanks;

  auto n_partition = new global_ordinal_type[numRanks+1];
  auto k_partition = new global_ordinal_type[numRanks+1];
  auto m_partition = new global_ordinal_type[numRanks+1];
  n_partition[0] = 0;
  k_partition[0] = 0;
  m_partition[0] = 0;

  for(int i = 1; i < numRanks + 1; ++i)
  {
    n_partition[i] = i*N_local;
    k_partition[i] = i*K_local;
    m_partition[i] = i*M_local;
  }

  int fail = 0;

  // Test dense matrix
  {
    if (rank==0)
      std::cout << "\nTesting hiopMatrixDenseRowMajor ...\n";
    // Matrix dimensions denoted by subscript
    // Distributed matrices (only distributed by columns):
    hiop::hiopMatrixDenseRowMajor A_kxm(K_local, M_global, m_partition, comm);
    hiop::hiopMatrixDenseRowMajor A_kxn(K_local, N_global, n_partition, comm);
    hiop::hiopMatrixDenseRowMajor A_mxk(M_local, K_global, k_partition, comm);
    hiop::hiopMatrixDenseRowMajor A_mxn(M_local, N_global, n_partition, comm);
    hiop::hiopMatrixDenseRowMajor A_nxm(N_local, M_global, m_partition, comm);
    // Try factory instead of constructor
    hiop::hiopMatrixDense* B_mxn = 
      hiop::LinearAlgebraFactory::createMatrixDense(M_local, N_global, n_partition, comm);
    hiop::hiopMatrixDense* A_mxn_extra_row =
      hiop::LinearAlgebraFactory::createMatrixDense(M_local, N_global, n_partition, comm, M_local+1);

    // Non-distributed matrices:
    hiop::hiopMatrixDenseRowMajor A_mxk_nodist(M_local, K_local);
    // Try factory instead of constructor
    hiop::hiopMatrixDense* A_mxm_nodist =
      hiop::LinearAlgebraFactory::createMatrixDense(M_local, M_local);
    hiop::hiopMatrixDenseRowMajor A_kxn_nodist(K_local, N_local);
    hiop::hiopMatrixDenseRowMajor A_kxm_nodist(K_local, M_local);
    hiop::hiopMatrixDenseRowMajor A_mxn_nodist(M_local, N_local);
    hiop::hiopMatrixDenseRowMajor A_nxn_nodist(N_local, N_local);

    // Vectors with shape of the form:
    // x_<size>_[non-distributed]
    //
    // Distributed vectors:
    hiop::hiopVectorPar x_n(N_global, n_partition, comm);

    // Non-distributed vectors
    hiop::hiopVectorPar x_n_nodist(N_local);
    hiop::hiopVectorPar x_m_nodist(M_local);

    hiop::tests::MatrixTestsDense test;

    fail += test.matrixSetToZero(A_mxn, rank);
    fail += test.matrixSetToConstant(A_mxn, rank);
    fail += test.matrixTimesVec(A_mxn, x_m_nodist, x_n, rank);
    fail += test.matrixTransTimesVec(A_mxn, x_m_nodist, x_n, rank);

    if(rank == 0)
    {
      // These functions are only meant to be called locally
      fail += test.matrixTimesMat(A_mxk_nodist, A_kxn_nodist, A_mxn_nodist);
      fail += test.matrixAddDiagonal(A_nxn_nodist, x_n_nodist);
      fail += test.matrixAddSubDiagonal(A_nxn_nodist, x_m_nodist);
      fail += test.matrixAddToSymDenseMatrixUpperTriangle(A_nxn_nodist, A_mxk_nodist);
      fail += test.matrixTransAddToSymDenseMatrixUpperTriangle(A_nxn_nodist, A_kxm_nodist);
      fail += test.matrixAddUpperTriangleToSymDenseMatrixUpperTriangle(A_nxn_nodist, *A_mxm_nodist);
#ifdef HIOP_DEEPCHECKS
      fail += test.matrixAssertSymmetry(A_nxn_nodist);
      fail += test.matrixOverwriteUpperTriangleWithLower(A_nxn_nodist);
      fail += test.matrixOverwriteLowerTriangleWithUpper(A_nxn_nodist);
#endif
      // Not part of hiopMatrix interface, specific to matrixTestsDense
      fail += test.matrixCopyBlockFromMatrix(*A_mxm_nodist, A_kxn_nodist);
      fail += test.matrixCopyFromMatrixBlock(A_kxn_nodist, *A_mxm_nodist);
    }

    fail += test.matrixTransTimesMat(A_mxk_nodist, A_kxn, A_mxn, rank);
    fail += test.matrixTimesMatTrans(A_mxn, A_mxk_nodist, A_kxn, rank);
    fail += test.matrixAddMatrix(A_mxn, *B_mxn, rank);
    fail += test.matrixMaxAbsValue(A_mxn, rank);
    fail += test.matrixIsFinite(A_mxn, rank);
    fail += test.matrixNumRows(A_mxn, M_local, rank); //<-- no row partitioning
    fail += test.matrixNumCols(A_mxn, N_global, rank);

    // specific to matrixTestsDense
    fail += test.matrixCopyFrom(A_mxn, *B_mxn, rank);

    fail += test.matrixAppendRow(*A_mxn_extra_row, x_n, rank);
    fail += test.matrixCopyRowsFrom(A_kxn, A_mxn, rank);
    fail += test.matrixCopyRowsFromSelect(A_mxn, A_kxn, rank);
    fail += test.matrixShiftRows(A_mxn, rank);
    fail += test.matrixReplaceRow(A_mxn, x_n, rank);
    fail += test.matrixGetRow(A_mxn, x_n, rank);

    // Delete test objects
    delete B_mxn;
    delete A_mxm_nodist;
    delete A_mxn_extra_row;
  }

  // Test RAJA dense matrix
#ifdef HIOP_USE_RAJA
  {
    if (rank==0)
      std::cout << "\nTesting hiopMatrixRajaDense ...\n";

    options.SetStringValue("mem_space", "device");
    hiop::LinearAlgebraFactory::set_mem_space(options.GetString("mem_space"));
    std::string mem_space = hiop::LinearAlgebraFactory::get_mem_space();

    // Matrix dimensions denoted by subscript
    // Distributed matrices (only distributed by columns):
    hiop::hiopMatrixRajaDense A_kxm(K_local, M_global, mem_space, m_partition, comm);
    hiop::hiopMatrixRajaDense A_kxn(K_local, N_global, mem_space, n_partition, comm);
    hiop::hiopMatrixRajaDense A_mxk(M_local, K_global, mem_space, k_partition, comm);
    hiop::hiopMatrixRajaDense A_mxn(M_local, N_global, mem_space, n_partition, comm);
    hiop::hiopMatrixRajaDense A_nxm(N_local, M_global, mem_space, m_partition, comm);
    // Try factory instead of constructor
    hiop::hiopMatrixDense* B_mxn =
      hiop::LinearAlgebraFactory::createMatrixDense(M_local, N_global, n_partition, comm);
    hiop::hiopMatrixDense* A_mxn_extra_row =
      hiop::LinearAlgebraFactory::createMatrixDense(M_local, N_global, n_partition, comm, M_local+1);

    // Non-distributed matrices:
    hiop::hiopMatrixRajaDense A_mxk_nodist(M_local, K_local, mem_space);
    // Try factory instead of constructor
    hiop::hiopMatrixDense* A_mxm_nodist =
      hiop::LinearAlgebraFactory::createMatrixDense(M_local, M_local);
    hiop::hiopMatrixRajaDense A_kxn_nodist(K_local, N_local, mem_space);
    hiop::hiopMatrixRajaDense A_kxm_nodist(K_local, M_local, mem_space);
    hiop::hiopMatrixRajaDense A_mxn_nodist(M_local, N_local, mem_space);
    hiop::hiopMatrixRajaDense A_nxn_nodist(N_local, N_local, mem_space);

    // Vectors with shape of the form:
    // x_<size>_[non-distributed]
    //
    // Distributed vectors:
    hiop::hiopVectorRajaPar x_n(N_global, mem_space, n_partition, comm);

    // Non-distributed vectors
    hiop::hiopVectorRajaPar x_n_nodist(N_local, mem_space);
    hiop::hiopVectorRajaPar x_m_nodist(M_local, mem_space);

    hiop::tests::MatrixTestsRajaDense test;

    fail += test.matrixSetToZero(A_mxn, rank);
    fail += test.matrixSetToConstant(A_mxn, rank);
    fail += test.matrixTimesVec(A_mxn, x_m_nodist, x_n, rank);
    fail += test.matrixTransTimesVec(A_mxn, x_m_nodist, x_n, rank);

    if(rank == 0)
    {
      // These methods are local
      fail += test.matrixTimesMat(A_mxk_nodist, A_kxn_nodist, A_mxn_nodist);
      fail += test.matrixAddDiagonal(A_nxn_nodist, x_n_nodist);
      fail += test.matrixAddSubDiagonal(A_nxn_nodist, x_m_nodist);
      fail += test.matrixAddToSymDenseMatrixUpperTriangle(A_nxn_nodist, A_mxk_nodist);
      fail += test.matrixTransAddToSymDenseMatrixUpperTriangle(A_nxn_nodist, A_kxm_nodist);
      fail += test.matrixAddUpperTriangleToSymDenseMatrixUpperTriangle(A_nxn_nodist, *A_mxm_nodist);
#ifdef HIOP_DEEPCHECKS
      fail += test.matrixAssertSymmetry(A_nxn_nodist);
      fail += test.matrixOverwriteUpperTriangleWithLower(A_nxn_nodist);
      fail += test.matrixOverwriteLowerTriangleWithUpper(A_nxn_nodist);
#endif
      // Not part of hiopMatrix interface, specific to matrixTestsDense
      fail += test.matrixCopyBlockFromMatrix(*A_mxm_nodist, A_kxn_nodist);
      fail += test.matrixCopyFromMatrixBlock(A_kxn_nodist, *A_mxm_nodist);
    }

    fail += test.matrixTransTimesMat(A_mxk_nodist, A_kxn, A_mxn, rank);
    fail += test.matrixTimesMatTrans(A_mxn, A_mxk_nodist, A_kxn, rank);
    fail += test.matrixAddMatrix(A_mxn, *B_mxn, rank);
    fail += test.matrixMaxAbsValue(A_mxn, rank);
    fail += test.matrixIsFinite(A_mxn, rank);
    fail += test.matrixNumRows(A_mxn, M_local, rank); //<- no row partitioning
    fail += test.matrixNumCols(A_mxn, N_global, rank);

    // specific to matrixTestsDense
    fail += test.matrixCopyFrom(A_mxn, *B_mxn, rank);

    fail += test.matrixAppendRow(*A_mxn_extra_row, x_n, rank);
    fail += test.matrixCopyRowsFrom(A_kxn, A_mxn, rank);
    fail += test.matrixCopyRowsFromSelect(A_mxn, A_kxn, rank);
    fail += test.matrixShiftRows(A_mxn, rank);
    fail += test.matrixReplaceRow(A_mxn, x_n, rank);
    fail += test.matrixGetRow(A_mxn, x_n, rank);

    // Delete test objects
    delete B_mxn;
    delete A_mxm_nodist;
    delete A_mxn_extra_row;

    // Set memory space back to default value
    options.SetStringValue("mem_space", "default");
  }
#endif

  if (rank == 0)
  {
    if(fail)
    {
      std::cout << "\n" << fail << " dense matrix tests failed\n\n";
    }
    else
    {
      std::cout << "\nAll dense matrix tests passed\n\n";
    }
  }

  delete[] m_partition;
  delete[] n_partition;
  delete[] k_partition;

#ifdef HIOP_USE_MPI
  MPI_Finalize();
#endif

  return fail;
}
