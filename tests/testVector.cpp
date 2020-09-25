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
 * @file testVector.cpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>, PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 *
 */
#include <iostream>
#include <assert.h>

// This header contains HiOp's MPI definitions
#include <hiopOptions.hpp>
#include <hiopLinAlgFactory.hpp>
#include <hiopVectorPar.hpp>
#include <hiopVectorRajaPar.hpp>
#include <hiopVectorIntSeq.hpp>
#include <hiopVectorIntRaja.hpp>

#include "LinAlg/vectorTestsPar.hpp"
#include "LinAlg/vectorTestsRajaPar.hpp"
#include "LinAlg/vectorTestsIntSeq.hpp"
#include "LinAlg/vectorTestsIntRaja.hpp"


/**
 * @brief Main body of vector implementation testing code.
 *
 * @todo The size of the vector should be passed on the command line.
 *
 * @pre All test functions should return the same boolean value on all ranks.
 *
 */
int main(int argc, char** argv)
{
  using hiop::tests::global_ordinal_type;

  int rank=0;
  int numRanks=1;
  MPI_Comm comm = MPI_COMM_SELF;

#ifdef HIOP_USE_MPI
  int err;
  err  = MPI_Init(&argc, &argv);         assert(MPI_SUCCESS == err);
  comm = MPI_COMM_WORLD;
  err  = MPI_Comm_rank(comm, &rank);     assert(MPI_SUCCESS == err);
  err  = MPI_Comm_size(comm, &numRanks); assert(MPI_SUCCESS == err);
  if(0 == rank && MPI_SUCCESS == err) std::cout << "Running MPI enabled tests ...\n";
#endif
  hiop::hiopOptions options;

  global_ordinal_type Nlocal = 1000;
  global_ordinal_type Mlocal = 500;
  global_ordinal_type Nglobal = Nlocal*numRanks;

  auto n_partition = new global_ordinal_type [numRanks + 1];
  auto m_partition = new global_ordinal_type [numRanks + 1];
  n_partition[0] = 0;
  m_partition[0] = 0;

  for(int i = 1; i < numRanks + 1; ++i)
  {
    n_partition[i] = i*Nlocal;
    m_partition[i] = i*Mlocal;
  }

  int fail = 0;

  // Test parallel vector
  {
    hiop::LinearAlgebraFactory::set_mem_space(options.GetString("mem_space"));
    hiop::hiopVectorPar x(Nglobal, n_partition, comm);
    hiop::hiopVectorPar y(Nglobal, n_partition, comm);
    hiop::hiopVectorPar z(Nglobal, n_partition, comm);
    // Try using factory instead of constructor
    hiop::hiopVector* a = hiop::LinearAlgebraFactory::createVector(Nglobal, n_partition, comm);
    hiop::hiopVectorPar b(Nglobal, n_partition, comm);

    // Allocate vectors for testing non-MPI kernels
    hiop::hiopVectorPar v_smaller(Mlocal);
    // Try using factory instead of constructor
    hiop::hiopVector* v = hiop::LinearAlgebraFactory::createVector(Nlocal);

    hiop::tests::VectorTestsPar test;

    fail += test.vectorGetSize(x, Nglobal, rank);
    fail += test.vectorSetToZero(x, rank);
    fail += test.vectorSetToConstant(x, rank);
    fail += test.vectorSetToConstant_w_patternSelect(x, y, rank);
    fail += test.vectorCopyFrom(x, y, rank);
    fail += test.vectorCopyTo(x, y, rank);

    if (rank == 0)
    {
      fail += test.vectorCopyFromStarting(*v, v_smaller);
      fail += test.vectorStartingAtCopyFromStartingAt(v_smaller, *v);
      fail += test.vectorCopyToStarting(*v, v_smaller);
      fail += test.vectorStartingAtCopyToStartingAt(*v, v_smaller);
    }

    fail += test.vectorSelectPattern(x, y, rank);
    fail += test.vectorScale(x, rank);
    fail += test.vectorComponentMult(x, y, rank);
    fail += test.vectorComponentDiv(x, y, rank);
    fail += test.vectorComponentDiv_p_selectPattern(x, y, z, rank);
    fail += test.vectorOnenorm(x, rank);
    fail += test.vectorTwonorm(x, rank);
    fail += test.vectorInfnorm(x, rank);

    fail += test.vectorAxpy(x, y, rank);
    fail += test.vectorAxzpy(x, y, z, rank);
    fail += test.vectorAxdzpy(x, y, z, rank);
    fail += test.vectorAxdzpy_w_patternSelect(x, y, z, *a, rank);

    fail += test.vectorAddConstant(x, rank);
    fail += test.vectorAddConstant_w_patternSelect(x, y, rank);
    fail += test.vectorDotProductWith(x, y, rank);
    fail += test.vectorNegate(x, rank);
    fail += test.vectorInvert(x, rank);
    fail += test.vectorLogBarrier(x, y, rank);
    fail += test.vectorAddLogBarrierGrad(x, y, z, rank);
    fail += test.vectorLinearDampingTerm(x, y, z, rank);

    fail += test.vectorAllPositive(x, rank);
    fail += test.vectorAllPositive_w_patternSelect(x, y, rank);

    fail += test.vectorMin(x, rank);
    fail += test.vectorProjectIntoBounds(x, y, z, *a, b, rank);
    fail += test.vectorFractionToTheBdry(x, y, rank);
    fail += test.vectorFractionToTheBdry_w_pattern(x, y, z, rank);

    fail += test.vectorMatchesPattern(x, y, rank);
    fail += test.vectorAdjustDuals_plh(x, y, z, *a, rank);

    if (rank == 0)
    {
      fail += test.vectorIsnan(*v);
      fail += test.vectorIsinf(*v);
      fail += test.vectorIsfinite(*v);
    }

    // Delete testing objects
    delete a;
    delete v;
  }

  // Test MPI+RAJA vector
  {
    if (rank == 0)
      std::cout << "\nTesting HiOp RAJA vector:\n";

    options.SetStringValue("mem_space", "device");
    hiop::LinearAlgebraFactory::set_mem_space(options.GetString("mem_space"));
    std::string mem_space = hiop::LinearAlgebraFactory::get_mem_space();

    hiop::hiopVectorRajaPar x(Nglobal, mem_space, n_partition, comm);
    hiop::hiopVectorRajaPar y(Nglobal, mem_space, n_partition, comm);
    hiop::hiopVectorRajaPar z(Nglobal, mem_space, n_partition, comm);
    // Try using factory instead of constructor
    hiop::hiopVector* a = hiop::LinearAlgebraFactory::createVector(Nglobal, n_partition, comm);
    hiop::hiopVectorRajaPar b(Nglobal, mem_space, n_partition, comm);
    hiop::tests::VectorTestsRajaPar test;

    // Allocate vectors for testing non-MPI kernels
    hiop::hiopVectorRajaPar v_smaller(Mlocal, mem_space); // Mlocal < Nlocal
    // Try using factory instead of constructor
    hiop::hiopVector* v = hiop::LinearAlgebraFactory::createVector(Nlocal);

    fail += test.vectorGetSize(x, Nglobal, rank);
    fail += test.vectorSetToZero(x, rank);
    fail += test.vectorSetToConstant(x, rank);
    fail += test.vectorSetToConstant_w_patternSelect(x, y, rank);
    fail += test.vectorCopyFrom(x, y, rank);
    fail += test.vectorCopyTo(x, y, rank);

    if (rank == 0)
    {
      fail += test.vectorCopyFromStarting(*v, v_smaller);
      fail += test.vectorStartingAtCopyFromStartingAt(v_smaller, *v);
      fail += test.vectorCopyToStarting(*v, v_smaller);
      fail += test.vectorStartingAtCopyToStartingAt(*v, v_smaller);
    }

    fail += test.vectorSelectPattern(x, y, rank);
    fail += test.vectorScale(x, rank);
    fail += test.vectorComponentMult(x, y, rank);
    fail += test.vectorComponentDiv(x, y, rank);
    fail += test.vectorComponentDiv_p_selectPattern(x, y, z, rank);
    fail += test.vectorOnenorm(x, rank);
    fail += test.vectorTwonorm(x, rank);
    fail += test.vectorInfnorm(x, rank);

    fail += test.vectorAxpy(x, y, rank);
    fail += test.vectorAxzpy(x, y, z, rank);
    fail += test.vectorAxdzpy(x, y, z, rank);
    fail += test.vectorAxdzpy_w_patternSelect(x, y, z, *a, rank);

    fail += test.vectorAddConstant(x, rank);
    fail += test.vectorAddConstant_w_patternSelect(x, y, rank);
    fail += test.vectorDotProductWith(x, y, rank);
    fail += test.vectorNegate(x, rank);
    fail += test.vectorInvert(x, rank);
    fail += test.vectorLogBarrier(x, y, rank);
    fail += test.vectorAddLogBarrierGrad(x, y, z, rank);
    fail += test.vectorLinearDampingTerm(x, y, z, rank);

    fail += test.vectorAllPositive(x, rank);
    fail += test.vectorAllPositive_w_patternSelect(x, y, rank);

    fail += test.vectorMin(x, rank);
    fail += test.vectorProjectIntoBounds(x, y, z, *a, b, rank);
    fail += test.vectorFractionToTheBdry(x, y, rank);
    fail += test.vectorFractionToTheBdry_w_pattern(x, y, z, rank);

    fail += test.vectorMatchesPattern(x, y, rank);
    fail += test.vectorAdjustDuals_plh(x, y, z, *a, rank);

    if (rank == 0)
    {
      fail += test.vectorIsnan(*v);
      fail += test.vectorIsinf(*v);
      fail += test.vectorIsfinite(*v);
    }

    // Delete testing objects
    delete a;
    delete v;

    // Set memory space back to default value
    options.SetStringValue("mem_space", "default");
  }

  // Test hiopVectorIntSeq
  if (rank == 0)
  {
    std::cout << "\nTesting HiOp sequential int vector:\n";

    options.SetStringValue("mem_space", "DEFAULT");
    hiop::LinearAlgebraFactory::set_mem_space(options.GetString("mem_space"));
    hiop::tests::VectorTestsIntSeq test;
    const int sz = 100;
    auto* x = hiop::LinearAlgebraFactory::createVectorInt(sz);
    fail += test.vectorSize(*x, sz);
    fail += test.vectorGetElement(*x);
    fail += test.vectorSetElement(*x);
  }

  // Test hiopVectorIntRaja
  if (rank == 0)
  {
    std::cout << "\nTesting HiOp RAJA int vector:\n";

    options.SetStringValue("mem_space", "DEVICE");
    hiop::LinearAlgebraFactory::set_mem_space(options.GetString("mem_space"));
    hiop::tests::VectorTestsIntRaja test;
    const int sz = 100;
    auto* x = hiop::LinearAlgebraFactory::createVectorInt(sz);
    fail += test.vectorSize(*x, sz);
    fail += test.vectorGetElement(*x);
    fail += test.vectorSetElement(*x);
    options.SetStringValue("mem_space", "DEFAULT");
  }

  if (rank == 0)
  {
    if(fail)
      std::cout << "\n" << fail << " vector tests failed!\n\n";
    else
      std::cout << "\nAll vector tests passed!\n\n";
  }

  delete[] m_partition;
  delete[] n_partition;

#ifdef HIOP_USE_MPI
  MPI_Finalize();
#endif

  return fail;
}
