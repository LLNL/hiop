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
 * @file hiopLinAlgFactory.cpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>, PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 * 
 */
#include <algorithm>
#include <iostream>

#include <hiop_defs.hpp>

#ifdef HIOP_USE_RAJA
#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>
#include <hiopVectorIntRaja.hpp>
#include <hiopVectorRajaPar.hpp>
#include <hiopMatrixRajaDense.hpp>
#include <hiopMatrixRajaSparseTriplet.hpp>
#endif // HIOP_USE_RAJA

#include <hiopVectorIntSeq.hpp>
#include <hiopVectorPar.hpp>
#include <hiopMatrixDenseRowMajor.hpp>
#include <hiopMatrixSparseTriplet.hpp>

#include "hiopLinAlgFactory.hpp"

using namespace hiop;

/**
 * @brief Method to create vector.
 * 
 * Creates legacy HiOp vector by default, RAJA vector when memory space
 * is specified.
 */
hiopVector* LinearAlgebraFactory::createVector(
  const long long& glob_n,
  long long* col_part,
  MPI_Comm comm)
{
  if(mem_space_ == "DEFAULT")
  {
    return new hiopVectorPar(glob_n, col_part, comm);
  }
  else
  {
#ifdef HIOP_USE_RAJA
    return new hiopVectorRajaPar(glob_n, mem_space_, col_part, comm);
#else
    assert(false && "requested memory space not available because Hiop was not"
           "built with RAJA support");
    return new hiopVectorPar(glob_n, col_part, comm);
#endif
  }
}

/**
 * @brief Method to create local int vector.
 * 
 * Creates int vector with operator new by default, RAJA vector when memory space
 * is specified.
 */
hiopVectorInt* LinearAlgebraFactory::createVectorInt(hiopInt size)
{
  if(mem_space_ == "DEFAULT")
  {
    return new hiopVectorIntSeq(size);
  }
  else
  {
#ifdef HIOP_USE_RAJA
    return new hiopVectorIntRaja(size, mem_space_);
#else
    assert(false && "requested memory space not available because Hiop was not"
           "built with RAJA support");
    return new hiopVectorIntSeq(size);
#endif
  }
}

/**
 * @brief Method to create matrix.
 * 
 * Creates legacy HiOp dense matrix by default, RAJA vector when memory space
 * is specified.
 */
hiopMatrixDense* LinearAlgebraFactory::createMatrixDense(
  const long long& m,
  const long long& glob_n,
  long long* col_part,
  MPI_Comm comm,
  const long long& m_max_alloc)
{
  if(mem_space_ == "DEFAULT")
  {
    return new hiopMatrixDenseRowMajor(m, glob_n, col_part, comm, m_max_alloc);
  }
  else
  {
#ifdef HIOP_USE_RAJA
    return new hiopMatrixRajaDense(m, glob_n, mem_space_, col_part, comm, m_max_alloc);
#else
    assert(false && "requested memory space not available because Hiop was not"
           "built with RAJA support");
    return new hiopMatrixDenseRowMajor(m, glob_n, col_part, comm, m_max_alloc);
#endif
  }
  
}

/**
 * @brief Creates an instance of a sparse matrix of the appropriate implementation
 * depending on the build.
 */
hiopMatrixSparse* LinearAlgebraFactory::createMatrixSparse(int rows, int cols, int nnz)
{
  if (mem_space_ == "DEFAULT")
  {
    return new hiopMatrixSparseTriplet(rows, cols, nnz);
  }
  else
  {
#ifdef HIOP_USE_RAJA
    return new hiopMatrixRajaSparseTriplet(rows, cols, nnz, mem_space_);
#else
    assert(false && "requested memory space not available because Hiop was not"
           "built with RAJA support");
    return new hiopMatrixSparseTriplet(rows, cols, nnz);
#endif
  }
}

/**
 * @brief Creates an instance of a symmetric sparse matrix of the appropriate
 * implementation depending on the build.
 */
hiopMatrixSparse* LinearAlgebraFactory::createMatrixSymSparse(int size, int nnz)
{
  if (mem_space_ == "DEFAULT")
  {
    return new hiopMatrixSymSparseTriplet(size, nnz);
  }
  else
  {
#ifdef HIOP_USE_RAJA
    return new hiopMatrixRajaSymSparseTriplet(size, nnz, mem_space_);
#else
    assert(false && "requested memory space not available because Hiop was not"
           "built with RAJA support");
    return new hiopMatrixSymSparseTriplet(size, nnz);
#endif
  }
}

/**
 * @brief Static method to create a raw C array
 */
double* LinearAlgebraFactory::createRawArray(int n)
{
  if (mem_space_ == "DEFAULT")
  {
    return new double[n];
  }
  else
  {
#ifdef HIOP_USE_RAJA
    auto& resmgr = umpire::ResourceManager::getInstance();
    umpire::Allocator al  = resmgr.getAllocator(mem_space_);
    return static_cast<double*>(al.allocate(n*sizeof(double)));
#else
    assert(false && "requested memory space not available because Hiop was not"
           "built with RAJA support");
    return new double[n];
#endif
  }
}

/**
 * @brief Static method to delete a raw C array
 */
void LinearAlgebraFactory::deleteRawArray(double* a)
{
  if (mem_space_ == "DEFAULT")
  {
    delete [] a;
  }
  else
  {
#ifdef HIOP_USE_RAJA
    auto& resmgr = umpire::ResourceManager::getInstance();
    umpire::Allocator al  = resmgr.getAllocator(mem_space_);
    al.deallocate(a);
#endif
  }
}


void LinearAlgebraFactory::set_mem_space(const std::string mem_space)
{
  mem_space_ = mem_space;
  // HiOp options turn all strings to lowercase. Umpire wants uppercase.
  transform(mem_space_.begin(), mem_space_.end(), mem_space_.begin(), ::toupper);
}

std::string LinearAlgebraFactory::mem_space_ = "DEFAULT";
