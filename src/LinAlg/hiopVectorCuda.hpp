// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp 
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause). 
// Please also read "Additional BSD Notice" below.
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
 * @file hiopVectorCuda.hpp
 *
 * @author Nai-Yuan Chiabg <chiang7@llnl.gov>, LLNL
 */
#ifndef HIOP_VECTOR_CUDA
#define HIOP_VECTOR_CUDA

#include <cstdio>
#include <string>
#include <cassert>
#include <cstring>

#include <hiopMPI.hpp>

#include "hiopVector.hpp"

#include "MathDeviceKernels.hpp"
#include "hiop_cusolver_defs.hpp"


namespace hiop
{

class hiopVectorCuda
{
public:
  hiopVectorCuda(const size_type& glob_n, std::string mem_space, index_type* col_part=NULL, MPI_Comm comm=MPI_COMM_SELF);
  virtual ~hiopVectorCuda();

  /// @brief Set all elements to zero.
  virtual void setToZero();
  /// @brief Set all elements to c
  virtual void setToConstant(double c);
  /// @brief Set all elements to random values uniformly distributed between `minv` and `maxv`.
  virtual void set_to_random_uniform(double minv, double maxv);
  /// @brief Set all elements that are not zero in ix to  c, and the rest to 0
  virtual void setToConstant_w_patternSelect(double c, const hiopVector& select);

  /// @brief Copy the elements of v
  virtual void copyFrom(const hiopVector& v );
  virtual void copyFrom(const double* v_local_data);
  virtual void copy_from_w_pattern(const hiopVector& src, const hiopVector& select);
  /// @brief Copy the 'n' elements of v starting at 'start_index_in_dest' in 'this'
  virtual void copyFromStarting(int start_index_in_dest, const double* v, int n);
  /// @brief Copy v_src into 'this' starting at start_index_in_dest in 'this'. */
  virtual void copyFromStarting(int start_index_in_dest, const hiopVector& v_src);
  /// @brief Copy the 'n' elements of v starting at 'start_index_in_v' into 'this'
  virtual void copy_from_starting_at(const double* v, int start_index_in_v, int n);

  /// @brief Copy from src the elements specified by the indices in index_in_src. 
  virtual void copy_from_indexes(const hiopVector& src, const hiopVectorInt& index_in_src) {assert(false&&"TODO");};

  /// @brief Copy from src the elements specified by the indices in index_in_src. 
  virtual void copy_from_indexes(const double* src, const hiopVectorInt& index_in_src) {assert(false&&"TODO");};

  ///  @brief Copy from 'v' starting at 'start_idx_src' to 'this' starting at 'start_idx_dest'
  virtual void startingAtCopyFromStartingAt(int start_idx_dest, const hiopVector& v, int start_idx_src);

  /// @brief Copy 'this' to double array, which is assumed to be at least of 'n_local_' size.
  virtual void copyTo(double* dest) const;
  /// @brief Copy 'this' to v starting at start_index in 'this'.
  virtual void copyToStarting(int start_index_in_src, hiopVector& v) const;
  /// @brief Copy 'this' to v starting at start_index in 'v'.
  virtual void copyToStarting(hiopVector& v, int start_index_in_dest) const;
  /// @brief Copy the entries in 'this' where corresponding 'ix' is nonzero, to v starting at start_index in 'v'.
  virtual void copyToStartingAt_w_pattern(hiopVector& v, int start_index_in_dest, const hiopVector& ix) const {assert(false&&"TODO");};

#if 0
  /// @brief Copy the entries in `c` and `d` to `this`, according to the mapping in `c_map` and `d_map`
  virtual void copy_from_two_vec_w_pattern(const hiopVector& c, 
                                           const hiopVectorInt& c_map, 
                                           const hiopVector& d, 
                                           const hiopVectorInt& d_map);

  /// @brief Copy the entries in `this` to `c` and `d`, according to the mapping `c_map` and `d_map`
  virtual void copy_to_two_vec_w_pattern(hiopVector& c, 
                                         const hiopVectorInt& c_map, 
                                         hiopVector& d, 
                                         const hiopVectorInt& d_map) const;
#endif



  /** @brief Return the two norm */
  virtual double twonorm() const;

#if 1
  /** @brief L1 norm on single rank */
  virtual double onenorm_local() const;
#endif 
  /** @brief Multiply the components of this by the components of v. */
  virtual void componentMult( const hiopVector& v );
  /** @brief Divide the components of this hiopVector by the components of v. */
  virtual void componentDiv ( const hiopVector& v ) = 0;

  /// @brief Scale each element of this  by the constant alpha
  virtual void scale(double alpha);
  /// @brief this += alpha * x
  virtual void axpy(double alpha, const hiopVector& x);
  /// @brief this += alpha * x, for the entries in 'this' where corresponding 'select' is nonzero.
  virtual void axpy_w_pattern(double alpha, const hiopVector& xvec, const hiopVector& select);

private:
  std::string mem_space_;
  MPI_Comm comm_;
  double* data_host_;
  double* data_dev_;

  size_type glob_il_, glob_iu_;
  size_type n_local_;

  /** needed for cuda **/
  cublasHandle_t handle_cublas_;
//  cublasStatus_t ret_cublas_;

  /** copy constructor, for internal/private use only (it doesn't copy the elements.) */
  hiopVectorCuda(const hiopVectorCuda&);

  // FIXME_NY
  // from hiopVector. remove these later
  size_type n_;
};

} // namespace hiop

#endif // HIOP_VECTOR_CUDA
