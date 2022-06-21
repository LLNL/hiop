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
 * @file hiopVectorRajaPar.cpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>, PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 * @author Jake K. Ryan <jake.ryan@pnnl.gov>, PNNL
 * @author Cameron Rutherford <cameron.rutherford@pnnl.gov>, PNNL
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 *
 */
#include "hiopVectorRajaPar.hpp"
#include "hiopVectorIntRaja.hpp"
#ifdef HIOP_USE_CUDA
#include "hiopUtilsCUDA.hpp"
#endif

#include <cmath>
#include <cstring> //for memcpy
#include <sstream>
#include <algorithm>
#include <cassert>

#include "hiop_blasdefs.hpp"

#include <limits>
#include <cstddef>

#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>

#include <RAJA/RAJA.hpp>
#include "hiop_raja_defs.hpp"


namespace hiop
{
// Define type aliases
using real_type = double;
using local_index_type = index_type;
using global_index_type = index_type;

// Define constants
static constexpr real_type zero = 0.0;
static constexpr real_type one  = 1.0;

hiopVectorRajaPar::hiopVectorRajaPar(
  const size_type& glob_n,
  std::string mem_space /* = "HOST" */,
  index_type* col_part /* = NULL */,
  MPI_Comm comm /* = MPI_COMM_NULL */)
  : hiopVector(),
    mem_space_(mem_space),
    comm_(comm)
{
  n_ = glob_n;

#ifdef HIOP_USE_MPI
  // if this is a serial vector, make sure it has a valid comm in the mpi case
  if(comm_ == MPI_COMM_NULL) 
    comm_ = MPI_COMM_SELF;
#endif

  int P = 0; 
  if(col_part)
  {
#ifdef HIOP_USE_MPI
    int ierr=MPI_Comm_rank(comm_, &P);  assert(ierr==MPI_SUCCESS);
#endif
    glob_il_ = col_part[P];
    glob_iu_ = col_part[P+1];
  } 
  else
  {
    glob_il_ = 0;
    glob_iu_ = n_;
  }
  n_local_ = glob_iu_ - glob_il_;

#ifndef HIOP_USE_GPU
  mem_space_ = "HOST"; // If no GPU support, fall back to host!
#endif

  auto& resmgr = umpire::ResourceManager::getInstance();
  umpire::Allocator devalloc  = resmgr.getAllocator(mem_space_);
  data_dev_ = static_cast<double*>(devalloc.allocate(n_local_*sizeof(double)));
  if(mem_space_ == "DEVICE")
  {
    // Create host mirror if the memory space is on device
    umpire::Allocator hostalloc = resmgr.getAllocator("HOST");
    data_host_ = static_cast<double*>(hostalloc.allocate(n_local_*sizeof(double)));
  }
  else
  {
    data_host_ = data_dev_;
  }
  //std::cout << "Memory space: " << mem_space_ << "\n";
}

hiopVectorRajaPar::hiopVectorRajaPar(const hiopVectorRajaPar& v)
  : hiopVector()
{
  n_local_ = v.n_local_;
  n_ = v.n_;
  glob_il_ = v.glob_il_;
  glob_iu_ = v.glob_iu_;
  comm_ = v.comm_;
  mem_space_ = v.mem_space_;

#ifndef HIOP_USE_GPU
  mem_space_ = "HOST"; // If no GPU support, fall back to host!
#endif

  // std::cout << "Memory space: " << mem_space_ << "\n";
  auto& resmgr = umpire::ResourceManager::getInstance();
  umpire::Allocator devalloc  = resmgr.getAllocator(mem_space_);
  data_dev_ = static_cast<double*>(devalloc.allocate(n_local_*sizeof(double)));
  if(mem_space_ == "DEVICE")
  {
    // Create host mirror if the memory space is on device
    umpire::Allocator hostalloc = resmgr.getAllocator("HOST");
    data_host_ = static_cast<double*>(hostalloc.allocate(n_local_*sizeof(double)));
  }
  else
  {
    data_host_ = data_dev_;
  }
}

hiopVectorRajaPar::~hiopVectorRajaPar()
{
  auto& resmgr = umpire::ResourceManager::getInstance();
  umpire::Allocator devalloc  = resmgr.getAllocator(mem_space_);
  if(data_dev_ != data_host_)
  {
    umpire::Allocator hostalloc = resmgr.getAllocator("HOST");
    hostalloc.deallocate(data_host_);
  }
  devalloc.deallocate(data_dev_);
  data_dev_  = nullptr;
  data_host_ = nullptr;
}

hiopVector* hiopVectorRajaPar::alloc_clone() const
{
  hiopVector* v = new hiopVectorRajaPar(*this); assert(v);
  return v;
}
hiopVector* hiopVectorRajaPar::new_copy () const
{
  hiopVector* v = new hiopVectorRajaPar(*this); assert(v);
  v->copyFrom(*this);
  return v;
}

//
// Compute kernels
//

/// Set all vector elements to zero
void hiopVectorRajaPar::setToZero()
{
  auto& rm = umpire::ResourceManager::getInstance();
  rm.memset(data_dev_, 0);
}

/// Set all vector elements to constant c
void hiopVectorRajaPar::setToConstant(double c)
{
  double* data = data_dev_;
  RAJA::forall< hiop_raja_exec >(RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      data[i] = c;
    });
}

void hiopVectorRajaPar::set_to_random_uniform(double minv, double maxv)
{
  double* data = data_dev_;
#ifdef HIOP_USE_CUDA
  hiop::cuda::array_random_uniform_kernel(n_local_, data);
#else
  assert(0&&"not yet");
#endif
}

/// Set selected elements to constant, zero otherwise
void hiopVectorRajaPar::setToConstant_w_patternSelect(double c, const hiopVector& select)
{
  const hiopVectorRajaPar& s = dynamic_cast<const hiopVectorRajaPar&>(select);
  const double* pattern = s.local_data_const();
  double* data = data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) {
      data[i] = pattern[i]*c;
    });
}

/**
 * @brief Copy data from vec to this vector
 * 
 * @param[in] vec - Vector from which to copy into `this`
 * 
 * @pre `vec` and `this` must have same partitioning.
 * @post Elements of `this` are overwritten with elements of `vec`
 */
void hiopVectorRajaPar::copyFrom(const hiopVector& vec)
{
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(vec);
  assert(n_local_ == v.n_local_);
  assert(glob_il_ == v.glob_il_);
  assert(glob_iu_ == v.glob_iu_);

  auto& rm = umpire::ResourceManager::getInstance();
  rm.copy(data_dev_, v.data_dev_);
}

/**
 * @brief Copy data from local_array to this vector
 * 
 * @param[in] local_array - A raw array from which to copy into `this`
 * 
 * @pre `local_array` is allocated by Umpire on device
 * @pre `local_array` must be of same size as the data block of `this`.
 * @post Elements of `this` are overwritten with elements of `local_array`.
 * 
 * @warning Method has no way to check for the size of `local_array`. May
 * read past the end of the array.
 * 
 * @warning Method casts away const from the `local_array`.
 * 
 * @warning Not tested - not part of the hiopVector interface.
 */
void hiopVectorRajaPar::copyFrom(const double* local_array)
{
  if(local_array) {
    auto& rm = umpire::ResourceManager::getInstance();
    double* data = const_cast<double*>(local_array);
    rm.copy(data_dev_, data, n_local_*sizeof(double));
  }
}

/// @brief Copy from vec the elements specified by the indices in index_in_src. 
void hiopVectorRajaPar::copy_from(const hiopVector& vec, const hiopVectorInt& index_in_src)
{
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(vec);
  const hiopVectorIntRaja& indexes = dynamic_cast<const hiopVectorIntRaja&>(index_in_src);
  size_type nv = v.get_local_size();

  assert(indexes.size() == n_local_);
  
  double* dd = data_dev_;
  double* vd = v.data_dev_;
  index_type* id = const_cast<index_type*>(indexes.local_data_const());

  RAJA::forall< hiop_raja_exec >(RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      assert(id[i]<nv);
      dd[i] = vd[id[i]];
    });
}

/// @brief Copy from vec the elements specified by the indices in index_in_src. 
void hiopVectorRajaPar::copy_from_w_pattern(const hiopVector& vec, const hiopVector& select)
{
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(vec);
  const hiopVectorRajaPar& ix = dynamic_cast<const hiopVectorRajaPar&>(select);

  assert(n_local_ == ix.n_local_);
  
  double* dd = data_dev_;
  double* vd = v.data_dev_;
  double* id = ix.data_dev_;

  RAJA::forall< hiop_raja_exec >(RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
       if(id[i] == one) {
         dd[i] = vd[i];
       }      
    });
}

/// @brief Copy from vec the elements specified by the indices in index_in_src
void hiopVectorRajaPar::copy_from(const double* vec, const hiopVectorInt& index_in_src)
{
  const hiopVectorIntRaja& indexes = dynamic_cast<const hiopVectorIntRaja&>(index_in_src);
  assert(indexes.size() == n_local_);
  
  assert(vec);
  double* dd = data_dev_;
  double* vd = const_cast<double*>(vec);
  index_type* id = const_cast<index_type*>(indexes.local_data_const());

  RAJA::forall< hiop_raja_exec >(RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      dd[i] = vd[id[i]];
    });
}

/// @brief Copy from vec the elements specified by the indices in index_in_src
void hiopVectorRajaPar::copy_from_indexes(const hiopVector& vv, const hiopVectorInt& index_in_src)
{
  const hiopVectorIntRaja& indexes = dynamic_cast<const hiopVectorIntRaja&>(index_in_src);
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(vv);

  assert(indexes.size() == n_local_);
  
  index_type* id = const_cast<index_type*>(indexes.local_data_const());
  double* dd = data_dev_;
  double* vd = v.data_dev_;

  size_type nv = v.get_local_size();
  
  RAJA::forall< hiop_raja_exec >(RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      assert(id[i]<nv);
      dd[i] = vd[id[i]];
    });  
}

/// @brief Copy from vec the elements specified by the indices in index_in_src
void hiopVectorRajaPar::copy_from_indexes(const double* vv, const hiopVectorInt& index_in_src)
{
  if(nullptr==vv) {
    return;
  }

  const hiopVectorIntRaja& indexes = dynamic_cast<const hiopVectorIntRaja&>(index_in_src);
  assert(indexes.size() == n_local_);
  index_type* id = const_cast<index_type*>(indexes.local_data_const());
  double* dd = data_dev_;
  
  RAJA::forall< hiop_raja_exec >(RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      dd[i] = vv[id[i]];
    }); 
}


/**
 * @brief Copy `nv` elements from array `v` to this vector starting from `start_index_in_this`
 * 
 * @param[in] start_index_in_this - position in this where to copy
 * @param[in] v  - a raw array from which to copy into `this`
 * @param[in] nv - how many elements of `v` to copy
 * 
 * @pre Size of `v` must be >= nv.
 * @pre start_index_in_this+nv <= n_local_
 * @pre `this` is not distributed
 * 
 * @warning Method casts away const from the `local_array`.
 */
void hiopVectorRajaPar::copyFromStarting(int start_index_in_this, const double* v, int nv)
{
  assert(start_index_in_this+nv <= n_local_);

  // If nothing to copy, return.  
  if(nv == 0)
    return;

  auto& rm = umpire::ResourceManager::getInstance();
  double* vv = const_cast<double*>(v); // <- cast away const
  rm.copy(data_dev_ + start_index_in_this, vv, nv*sizeof(double));
}

/**
 * @brief Copy `vec` to this vector starting from `start_index` in `this`.
 * 
 * @param[in] start_index - position in `this` where to copy
 * @param[in] src - the source vector from which to copy into `this`
 * 
 * @pre Size of `src` must be >= nv.
 * @pre start_index + src.n_local_ <= n_local_
 * @pre `this` is not distributed
 */
void hiopVectorRajaPar::copyFromStarting(int start_index, const hiopVector& src)
{
#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == n_ && "are you sure you want to call this?");
#endif
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(src);
  assert(start_index + v.n_local_ <= n_local_);

  // If there is nothing to copy, return.
  if(v.n_local_ == 0)
    return;

  auto& rm = umpire::ResourceManager::getInstance();
  double* vdata = const_cast<double*>(v.data_dev_); // scary:
  rm.copy(this->data_dev_ + start_index, vdata, v.n_local_*sizeof(double));
}

/**
 * @brief Copy `nv` elements from `start_index_in_v` at array `v` to this vector
 *
 * @param[in] start_index_in_v - position in v
 * @param[in] v  - a raw array from which to copy into `this`
 * @param[in] nv - how many elements of `v` to copy
 *
 * @pre Size of `v` must be >= nv.
 * @pre start_index_in_v+nv <= size of 'v'
 * @pre `this` is not distributed
 *
 * @warning Method casts away const from the `local_array`.
 */
void hiopVectorRajaPar::copy_from_starting_at(const double* v, int start_index_in_v, int nv)
{
  // If nothing to copy, return.
  if(nv == 0)
    return;

  auto& rm = umpire::ResourceManager::getInstance();
  double* vv = const_cast<double*>(v); // <- cast away const
  rm.copy(data_dev_, vv + start_index_in_v, nv*sizeof(double));
}

/**
 * @brief Copy from `vec_src` starting at `start_idx_src` into
 * `this` vector starting at `start_idx_dest`.
 * 
 * @pre `vec_src` and `this` are not distributed.
 * @pre `start_idx_dest` + `howManyToCopySrc` <= `n_local_`
 * @pre `start_idx_src` + `howManyToCopySrc` <= `vec_src.n_local_`
 * @post Elements of `vec_src` are unchanged.
 * @post All elements of `this` starting from `start_idx_dest` are overwritten
 * 
 * @todo Implentation differs from CPU - check with upstream what is correct!
 */
void hiopVectorRajaPar::startingAtCopyFromStartingAt(
  int start_idx_dest,
  const hiopVector& vec_src,
  int start_idx_src)
{
#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == n_ && "are you sure you want to call this?");
#endif
  assert((start_idx_dest >= 0 && start_idx_dest < this->n_local_) || this->n_local_==0);
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(vec_src);
  assert((start_idx_src >=0 && start_idx_src < v.n_local_) || v.n_local_==0 || v.n_local_==start_idx_src);

  size_type howManyToCopyDest = this->n_local_ - start_idx_dest;

#ifndef NDEBUG
  const size_type howManyToCopySrc = v.n_local_-start_idx_src;
  assert(howManyToCopyDest <= howManyToCopySrc);
#endif

  if(howManyToCopyDest == 0) {
    return;
  }

  auto& rm = umpire::ResourceManager::getInstance();
  rm.copy(this->data_dev_ + start_idx_dest, 
          v.data_dev_ + start_idx_src, 
          howManyToCopyDest*sizeof(double));
}

/**
 * @brief Copy to `vec` elements of `this` vector starting from `start_index`.
 * 
 * @param[in] start_index - position in `this` from where to copy
 * @param[out] dst - the destination vector where to copy elements of `this`
 * 
 * @pre start_index + dst.n_local_ <= n_local_
 * @pre `this` and `dst` are not distributed
 */
void hiopVectorRajaPar::copyToStarting(int start_index, hiopVector& dst) const
{
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(dst);

#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == n_ && "are you sure you want to call this?");
#endif
  assert(start_index + v.n_local_ <= n_local_);

  // If nowhere to copy, return.
  if(v.n_local_ == 0)
    return;

  auto& rm = umpire::ResourceManager::getInstance();
  rm.copy(v.data_dev_, this->data_dev_ + start_index, v.n_local_*sizeof(double));
}

/**
 * @brief Copy elements of `this` vector to `vec` starting at `start_index`.
 * 
 * @param[out] vec - a vector where to copy elements of `this`
 * @param[in] start_index_in_dest - position in `vec` where to copy
 * 
 * @pre start_index_in_dest + vec.n_local_ <= n_local_
 * @pre `this` and `vec` are not distributed
 */
void hiopVectorRajaPar::copyToStarting(hiopVector& vec, int start_index_in_dest) const
{
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(vec);
  assert(start_index_in_dest+n_local_ <= v.n_local_);

  // If there is nothing to copy, return.
  if(n_local_ == 0)
    return;

  auto& rm = umpire::ResourceManager::getInstance();
  rm.copy(v.data_dev_ + start_index_in_dest, this->data_dev_, this->n_local_*sizeof(double));
}

void hiopVectorRajaPar::copyToStartingAt_w_pattern(hiopVector& vec, int start_index_in_dest, const hiopVector& select) const
{
#if 0  
  if(n_local_ == 0)
    return;
 
  hiopVectorRajaPar& v = dynamic_cast<hiopVectorRajaPar&>(vec);
  const hiopVectorRajaPar& ix= dynamic_cast<const hiopVectorRajaPar&>(select);
  assert(n_local_ == ix.n_local_);
  
  int find_nnz = 0;
  double* dd = data_dev_;
  double* vd = v.data_dev_;
  double* id = ix.data_dev_;
  
  RAJA::ReduceSum< hiop_raja_reduce, double > sum(zero);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    [&](RAJA::Index_type i)
    {
      assert(id[i] == zero || id[i] == one);
      if(id[i] == one){
        vd[start_index_in_dest+find_nnz] = dd[i];
        find_nnz++;
      }
    });
#else
  assert(false && "not needed / implemented");
#endif    
}

/* copy 'c' and `d` into `this`, according to the map 'c_map` and `d_map`, respectively.
*  e.g., this[c_map[i]] = c[i];
*
*  @pre the size of `this` = the size of `c` + the size of `d`.
*  @pre `c_map` \Union `d_map` = {0, ..., size_of_this_vec-1}
*/
void hiopVectorRajaPar::copy_from_two_vec_w_pattern(const hiopVector& c,
                                                    const hiopVectorInt& c_map,
                                                    const hiopVector& d,
                                                    const hiopVectorInt& d_map)
{
  const hiopVectorRajaPar& v1 = dynamic_cast<const hiopVectorRajaPar&>(c);
  const hiopVectorRajaPar& v2 = dynamic_cast<const hiopVectorRajaPar&>(d);
  const hiopVectorIntRaja& ix1 = dynamic_cast<const hiopVectorIntRaja&>(c_map);
  const hiopVectorIntRaja& ix2 = dynamic_cast<const hiopVectorIntRaja&>(d_map);
  
  size_type n1_local = v1.n_local_;
  size_type n2_local = v2.n_local_;

#ifdef HIOP_DEEPCHECKS
  assert(n1_local + n2_local == n_local_);
  assert(n_local_ == ix1.size() + ix2.size());
#endif
  double*   dd = data_dev_;
  double*  vd1 = v1.data_dev_;
  double*  vd2 = v2.data_dev_;

  const index_type* id1 = ix1.local_data_const();
  const index_type* id2 = ix2.local_data_const();
  
  int n1_local_int = (int) n1_local;
  int n2_local_int = (int) n2_local;

  RAJA::forall< hiop_raja_exec >(
    RAJA::RangeSegment(0, n1_local_int),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      int idx = id1[i];
      dd[idx] = vd1[i];
    }
  );

  RAJA::forall< hiop_raja_exec >(
    RAJA::RangeSegment(0, n2_local_int),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      int idx = id2[i];
      dd[idx] = vd2[i];
    }
  );
}

/* split `this` to `c` and `d`, according to the map 'c_map` and `d_map`, respectively.
*
*  @pre the size of `this` = the size of `c` + the size of `d`.
*  @pre `c_map` \Union `d_map` = {0, ..., size_of_this_vec-1}
*/
void hiopVectorRajaPar::copy_to_two_vec_w_pattern(hiopVector& c,
                                                  const hiopVectorInt& c_map,
                                                  hiopVector& d,
                                                  const hiopVectorInt& d_map) const
{
  const hiopVectorRajaPar& v1 = dynamic_cast<const hiopVectorRajaPar&>(c);
  const hiopVectorRajaPar& v2 = dynamic_cast<const hiopVectorRajaPar&>(d);
  const hiopVectorIntRaja& ix1 = dynamic_cast<const hiopVectorIntRaja&>(c_map);
  const hiopVectorIntRaja& ix2 = dynamic_cast<const hiopVectorIntRaja&>(d_map);
  
  size_type n1_local = v1.n_local_;
  size_type n2_local = v2.n_local_;

#ifdef HIOP_DEEPCHECKS
  assert(n1_local + n2_local == n_local_);
  assert(n_local_ == ix1.size() + ix2.size());
#endif
  double*   dd = data_dev_;
  double*  vd1 = v1.data_dev_;
  double*  vd2 = v2.data_dev_;
  const index_type* id1 = ix1.local_data_const();
  const index_type* id2 = ix2.local_data_const();
  
  RAJA::forall< hiop_raja_exec >(
    RAJA::RangeSegment(0, (int)n1_local),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      int idx = id1[i];
      vd1[i] = dd[idx];
    }
  );

  RAJA::forall< hiop_raja_exec >(
    RAJA::RangeSegment(0, (int)n2_local),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      int idx = id2[i];
      vd2[i] = dd[idx];
    }
  );                                           
}

/**
 * @brief Copy elements of `this` vector to `destination` with offsets.
 * 
 * Copy `this` (source) starting at `start_idx_in_src` to `destination` 
 * starting at index 'int start_idx_dest'. If num_elems>=0, 'num_elems' will be copied; 
 * 
 * @param[out] vec - a vector where to copy elements of `this`
 * @param[in] start_idx_in_src - position in `vec` where to copy
 * 
 * @pre start_idx_in_src <= n_local_
 * @pre start_idx_dest   <= destination.n_local_
 * @pre `this` and `destination` are not distributed
 * @post If num_elems >= 0, `num_elems` will be copied
 * @post If num_elems < 0, elements will be copied till the end of
 * either source (`this`) or `destination` is reached
 */
void hiopVectorRajaPar::startingAtCopyToStartingAt(
  int start_idx_in_src, 
  hiopVector& destination, 
  int start_idx_dest, 
  int num_elems /* = -1 */) const
{

#ifdef HIOP_DEEPCHECKS
  assert(n_local_==n_ && "only for local/non-distributed vectors");
#endif  

  const hiopVectorRajaPar& dest = dynamic_cast<hiopVectorRajaPar&>(destination);

  assert(start_idx_in_src >= 0 && start_idx_in_src <= this->n_local_);
  assert(start_idx_dest   >= 0 && start_idx_dest   <= dest.n_local_);

#ifndef NDEBUG  
  if(start_idx_dest==dest.n_local_ || start_idx_in_src==this->n_local_) assert((num_elems==-1 || num_elems==0));
#endif

  if(num_elems<0)
  {
    num_elems = std::min(this->n_local_ - start_idx_in_src, dest.n_local_ - start_idx_dest);
  } 
  else
  {
    assert(num_elems+start_idx_in_src <= this->n_local_);
    assert(num_elems+start_idx_dest   <= dest.n_local_);
    //make sure everything stays within bounds (in release)
    num_elems = std::min(num_elems, (int)this->n_local_-start_idx_in_src);
    num_elems = std::min(num_elems, (int)dest.n_local_-start_idx_dest);
  }

  if(num_elems == 0)
    return;
  auto& rm = umpire::ResourceManager::getInstance();
  rm.copy(dest.data_dev_ + start_idx_dest, this->data_dev_ + start_idx_in_src, num_elems*sizeof(double));
}

void hiopVectorRajaPar::startingAtCopyToStartingAt_w_pattern(index_type start_idx_in_src,
                                                             hiopVector& destination,
                                                             index_type start_idx_dest,
                                                             const hiopVector& selec_dest,
                                                             size_type num_elems/*=-1*/) const
{
#if 0  
  hiopVectorRajaPar& dest = dynamic_cast<hiopVectorRajaPar&>(destination);
  const hiopVectorRajaPar& ix = dynamic_cast<const hiopVectorRajaPar&>(selec_dest);
    
  assert(start_idx_in_src >= 0 && start_idx_in_src <= this->n_local_);
  assert(start_idx_dest   >= 0 && start_idx_dest   <= dest.n_local_);
    
  if(num_elems<0)
  {
    num_elems = std::min(this->n_local_ - start_idx_in_src, dest.n_local_ - start_idx_dest);
  }
  else
  {
    assert(num_elems+start_idx_in_src <= this->n_local_);
    assert(num_elems+start_idx_dest   <= dest.n_local_);
    //make sure everything stays within bounds (in release)
    num_elems = std::min(num_elems, (int)this->n_local_-start_idx_in_src);
    num_elems = std::min(num_elems, (int)dest.n_local_-start_idx_dest);
  }
      
  int find_nnz = 0;
  double* dd = data_dev_;
  double* vd = dest.data_dev_;
  double* id = ix.data_dev_;

  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    [&](RAJA::Index_type i)
    {
      assert(id[i] == zero || id[i] == one);
      if(id[i] == one && find_nnz<num_elems)
        vd[start_idx_dest+find_nnz] = dd[ start_idx_in_src + (find_nnz++)];
    });
#else
  assert(false && "not needed / implemented");
#endif
}
 
 /**
 * @brief Copy `this` vector local data to `dest` buffer.
 * 
 * @param[out] dest - destination buffer where to copy vector data
 * 
 * @pre Size of `dest` must be >= n_local_
 * @post `this` is not modified
 */
void hiopVectorRajaPar::copyTo(double* dest) const
{
  auto& rm = umpire::ResourceManager::getInstance();
  rm.copy(dest, this->data_dev_, n_local_*sizeof(double));
}

/**
 * @brief L2 vector norm.
 * 
 * @post `this` is not modified
 * 
 * @todo Consider implementing with BLAS call (<D>NRM2).
 */
double hiopVectorRajaPar::twonorm() const
{
  double* self_dev = data_dev_;
  RAJA::ReduceSum<hiop_raja_reduce, double> sum(0.0);
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      sum += self_dev[i] * self_dev[i];
    });
  double nrm = sum.get();

#ifdef HIOP_USE_MPI
  double nrm_global;
  int ierr = MPI_Allreduce(&nrm, &nrm_global, 1, MPI_DOUBLE, MPI_SUM, comm_);
  assert(MPI_SUCCESS == ierr);
  return std::sqrt(nrm_global);
#endif  
  return std::sqrt(nrm);
}

/**
 * @brief scalar (dot) product.
 * 
 * @param[in] vec - vector which is scalar-multiplied to `this`.
 * 
 * @pre `vec` has same size and partitioning as `this`.
 * @post `this` and `vec` are not modified.
 * 
 * @todo Consider implementing with BLAS call (<D>DOT).
 */
double hiopVectorRajaPar::dotProductWith( const hiopVector& vec) const
{
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(vec);
  assert(n_local_ == v.n_local_);

  double* dd = data_dev_;
  double* vd = v.data_dev_;
  RAJA::ReduceSum<hiop_raja_reduce, double> dot(0.0);
  RAJA::forall<hiop_raja_exec>( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) {
      dot += dd[i] * vd[i];
    });
  double dotprod = dot.get();

#ifdef HIOP_USE_MPI
  double dotprodG;
  int ierr = MPI_Allreduce(&dotprod, &dotprodG, 1, MPI_DOUBLE, MPI_SUM, comm_);
  assert(MPI_SUCCESS==ierr);
  dotprod=dotprodG;
#endif

  return dotprod;
}

/**
 * @brief L-infinity (max) vector norm.
 * 
 * @post `this` is not modified
 * 
 */
double hiopVectorRajaPar::infnorm() const
{
  double nrm = infnorm_local();
#ifdef HIOP_USE_MPI
  double nrm_global;
  int ierr = MPI_Allreduce(&nrm, &nrm_global, 1, MPI_DOUBLE, MPI_MAX, comm_);
  assert(MPI_SUCCESS==ierr);
  return nrm_global;
#endif

  return nrm;
}

/**
 * @brief Local L-infinity (max) vector norm.
 * 
 * @pre  `this` is not empty vector
 * @post `this` is not modified
 * 
 */
double hiopVectorRajaPar::infnorm_local() const
{
  assert(n_local_ >= 0);
  double* data = data_dev_;
  RAJA::ReduceMax< hiop_raja_reduce, double > norm(0.0);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      norm.max(fabs(data[i]));
    });
  return norm.get();
}

/**
 * @brief 1-norm of `this` vector.
 * 
 * @post `this` is not modified
 * 
 */
double hiopVectorRajaPar::onenorm() const
{
  double norm1 = onenorm_local();
#ifdef HIOP_USE_MPI
  double nrm1_global;
  int ierr = MPI_Allreduce(&norm1, &nrm1_global, 1, MPI_DOUBLE, MPI_SUM, comm_); assert(MPI_SUCCESS==ierr);
  return nrm1_global;
#endif
  return norm1;
}

/**
 * @brief Local 1-norm of `this` vector.
 * 
 * @pre  `this` is not empty vector
 * @post `this` is not modified
 * 
 */
double hiopVectorRajaPar::onenorm_local() const
{
  double* data = data_dev_;
  RAJA::ReduceSum< hiop_raja_reduce, double > sum(0.0);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      sum += fabs(data[i]);
    });
  return sum.get();
}

/**
 * @brief Multiply `this` by `vec` elementwise and store result in `this`.
 * 
 * @pre  `this` and `vec` have same partitioning.
 * @post `vec` is not modified
 * 
 */
void hiopVectorRajaPar::componentMult(const hiopVector& vec)
{
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(vec);
  assert(n_local_ == v.n_local_);
  double* dd = data_dev_;
  double* vd = v.data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      dd[i] *= vd[i];
    });
}

/**
 * @brief Divide `this` vector elemenwise in-place by `vec`. 
 * 
 * @pre `this` and `vec` have same partitioning.
 * @pre vec[i] != 0 forall i
 * @post `vec` is not modified
 * 
 */
void hiopVectorRajaPar::componentDiv (const hiopVector& vec)
{
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(vec);
  assert(n_local_ == v.n_local_);
  double* dd = data_dev_;
  double* vd = v.data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      dd[i] /= vd[i];
    });
}

/**
 * @brief Divide `this` vector elemenwise in-place by `vec`
 * with pattern selection. 
 * 
 * @pre `this`, `select` and `vec` have same partitioning.
 * @pre vec[i] != 0 when select[i] = 1
 * @post `vec` and `select` are not modified
 * 
 */
void hiopVectorRajaPar::componentDiv_w_selectPattern( const hiopVector& vec, const hiopVector& select)
{
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(vec);
  const hiopVectorRajaPar& ix= dynamic_cast<const hiopVectorRajaPar&>(select);
#ifdef HIOP_DEEPCHECKS
  assert(v.n_local_ == n_local_);
  assert(n_local_ == ix.n_local_);
#endif
  double* dd = data_dev_;
  double* vd = v.data_dev_;
  double* id = ix.data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      assert(id[i] == zero || id[i] == one);
      if(id[i] == zero)
        dd[i] = zero;
      else  
        dd[i] /= vd[i];
    });
}

/**
 * @brief Set `this` vector elemenwise to the minimum of itself and the given `constant`
 */
void hiopVectorRajaPar::component_min(const double constant)
{
  double* dd = data_dev_;
  RAJA::forall< hiop_raja_exec >(
    RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      if(dd[i]>constant) {
        dd[i] = constant;
      }      
    }
  );
}

/**
 * @brief Set `this` vector elemenwise to the minimum of itself and the corresponding component of 'vec'.
 * 
 * @pre `this` and `vec` have same partitioning.
 * @post `vec` is not modified
 * 
 */
void hiopVectorRajaPar::component_min(const hiopVector& vec)
{
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(vec);
  assert(n_local_ == v.n_local_);
  double* dd = data_dev_;
  double* vd = v.data_dev_;
  RAJA::forall< hiop_raja_exec >(
    RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      if(dd[i]>vd[i]) {
        dd[i] = vd[i];
      } 
    }
  );
}

/**
 * @brief Set `this` vector elemenwise to the maximum of itself and the given `constant`
 */
void hiopVectorRajaPar::component_max(const double constant)
{
  double* dd = data_dev_;
  RAJA::forall< hiop_raja_exec >(
    RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      if(dd[i]<constant) {
        dd[i] = constant;
      }      
    }
  );
}

/**
 * @brief Set `this` vector elemenwise to the maximum of itself and the corresponding component of 'vec'.
 * 
 * @pre `this` and `vec` have same partitioning.
 * @post `vec` is not modified
 * 
 */
void hiopVectorRajaPar::component_max(const hiopVector& vec)
{
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(vec);
  assert(n_local_ == v.n_local_);
  double* dd = data_dev_;
  double* vd = v.data_dev_;
  RAJA::forall< hiop_raja_exec >(
    RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      if(dd[i]<vd[i]) {
        dd[i] = vd[i];
      } 
    }
  );
}

/**
 * @brief Set each component to its absolute value
 */
void hiopVectorRajaPar::component_abs ()
{
  double* dd = data_dev_;
  RAJA::forall< hiop_raja_exec >(
    RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      dd[i] = fabs(dd[i]);
    }
  );
}

/**
 * @brief Set each component to its absolute value
 */
void hiopVectorRajaPar::component_sgn ()
{
  double* dd = data_dev_;
  RAJA::forall< hiop_raja_exec >(
    RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      int sign = (0.0 < dd[i]) - (dd[i] < 0.0);
      dd[i] = static_cast<double>(sign);      
    }
  );
}

/**
 * @brief compute square root of each element
 * @pre all the elements are non-negative
 */
void hiopVectorRajaPar::component_sqrt()
{
  double* dd = data_dev_;
  RAJA::forall< hiop_raja_exec >(
    RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      dd[i] = sqrt(dd[i]);
    }
  );
}

/**
 * @brief Scale `this` vector by `c` 
 * 
 * @note Consider implementing with BLAS call (<D>SCAL)
 */
void hiopVectorRajaPar::scale(double c)
{
  if(1.0==c)
    return;
  
  double* data = data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      data[i] *= c;
    });
}

/**
 * @brief Implementation of AXPY kernel 
 * 
 * @pre `this` and `xvec` have same partitioning.
 * @post `xvec` is not modified
 * 
 * @note Consider implementing with BLAS call (<D>AXPY)
 */
void hiopVectorRajaPar::axpy(double alpha, const hiopVector& xvec)
{
  const hiopVectorRajaPar& x = dynamic_cast<const hiopVectorRajaPar&>(xvec);
  
  double* yd = data_dev_;
  double* xd = x.data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      // y := a * x + y
      yd[i] = alpha * xd[i] + yd[i];
    });
}

/// @brief Performs axpy, this += alpha*x, on the indexes in this specified by i.
void hiopVectorRajaPar::axpy(double alpha, const hiopVector& xvec, const hiopVectorInt& i)
{
  const hiopVectorRajaPar& x = dynamic_cast<const hiopVectorRajaPar&>(xvec);
  const hiopVectorIntRaja& idxs = dynamic_cast<const hiopVectorIntRaja&>(i);

  assert(x.get_size()==i.size());
  assert(x.get_local_size()==i.size());
  assert(i.size()<=n_local_);
  
  double* dd = data_dev_;
  double* xd = const_cast<double*>(x.data_dev_);
  index_type* id = const_cast<index_type*>(idxs.local_data_const());
  auto tmp_n_local = n_local_;

  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      assert(id[i]<tmp_n_local);
      // y := a * x + y
      dd[id[i]] = alpha * xd[i] + dd[id[i]];
    });
}

/// @brief Performs axpy, this += alpha*x, for selected entries
void hiopVectorRajaPar::axpy_w_pattern(double alpha, const hiopVector& xvec, const hiopVector& select)
{
  const hiopVectorRajaPar& x = dynamic_cast<const hiopVectorRajaPar&>(xvec);
  const hiopVectorRajaPar& sel = dynamic_cast<const hiopVectorRajaPar&>(select);
#ifdef HIOP_DEEPCHECKS
  assert(x.n_local_ == sel.n_local_);
  assert(  n_local_ == sel.n_local_);
#endif  
  double *dd       = data_dev_;
  const double *xd = x.local_data_const();
  const double *id = sel.local_data_const();
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      dd[i] += alpha * xd[i] * id[i];        
    });
}

/**
 * @brief this[i] += alpha*x[i]*z[i] forall i
 * 
 * @pre `this`, `xvec` and `zvec` have same partitioning.
 * @post `xvec` and `zvec` are not modified
 */
void hiopVectorRajaPar::axzpy(double alpha, const hiopVector& xvec, const hiopVector& zvec)
{
  const hiopVectorRajaPar& x = dynamic_cast<const hiopVectorRajaPar&>(xvec);
  const hiopVectorRajaPar& z = dynamic_cast<const hiopVectorRajaPar&>(zvec);
#ifdef HIOP_DEEPCHECKS
  assert(x.n_local_ == z.n_local_);
  assert(  n_local_ == z.n_local_);
#endif  
  double *dd       = data_dev_;
  const double *xd = x.local_data_const();
  const double *zd = z.local_data_const();
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      dd[i] += alpha*xd[i]*zd[i];
    });
}

/**
 * @brief this[i] += alpha*x[i]/z[i] forall i
 * 
 * @pre `this`, `xvec` and `zvec` have same partitioning.
 * @pre zvec[i] != 0 forall i
 * @post `xvec` and `zvec` are not modified
 */
void hiopVectorRajaPar::axdzpy(double alpha, const hiopVector& xvec, const hiopVector& zvec)
{
  const hiopVectorRajaPar& x = dynamic_cast<const hiopVectorRajaPar&>(xvec);
  const hiopVectorRajaPar& z = dynamic_cast<const hiopVectorRajaPar&>(zvec);
#ifdef HIOP_DEEPCHECKS
  assert(x.n_local_==z.n_local_);
  assert(  n_local_==z.n_local_);
#endif  
  double *yd       = data_dev_;
  const double *xd = x.local_data_const();
  const double *zd = z.local_data_const();
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      yd[i] += alpha*xd[i]/zd[i];
    });
}

/**
 * @brief this[i] += alpha*x[i]/z[i] forall i with pattern selection
 * 
 * @pre `this`, `xvec`, `zvec` and `select` have same partitioning.
 * @pre zvec[i] != 0 when select[i] = 1
 * @post `xvec`, `zvec` and `select` are not modified
 */
void hiopVectorRajaPar::axdzpy_w_pattern( 
  double alpha,
  const hiopVector& xvec, 
  const hiopVector& zvec,
  const hiopVector& select)
{
  const hiopVectorRajaPar& x = dynamic_cast<const hiopVectorRajaPar&>(xvec);
  const hiopVectorRajaPar& z = dynamic_cast<const hiopVectorRajaPar&>(zvec);
  const hiopVectorRajaPar& sel = dynamic_cast<const hiopVectorRajaPar&>(select);
#ifdef HIOP_DEEPCHECKS
  assert(x.n_local_==z.n_local_);
  assert(  n_local_==z.n_local_);
#endif  
  double* yd = data_dev_;
  const double* xd = x.local_data_const();
  const double* zd = z.local_data_const(); 
  const double* id = sel.local_data_const();
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) 
    {
      assert(id[i] == one || id[i] == zero);
      if(id[i] == one)
        yd[i] += alpha * xd[i] / zd[i];
    });
}

/**
 * @brief this[i] += c forall i
 * 
 */
void hiopVectorRajaPar::addConstant(double c)
{
  double *yd = data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      yd[i] += c;
    });
}

/**
 * @brief this[i] += c forall i with pattern selection
 * 
 * @pre `this` and `select` have same partitioning.
 * @post `select` is not modified
 */
void  hiopVectorRajaPar::addConstant_w_patternSelect(double c, const hiopVector& select)
{
  const hiopVectorRajaPar& sel = dynamic_cast<const hiopVectorRajaPar&>(select);
  assert(this->n_local_ == sel.n_local_);
  double *data = data_dev_;
  const double *id = sel.local_data_const();
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      assert(id[i] == one || id[i] == zero);
      data[i] += id[i]*c;
    });
}

/// Find minimum vector element
double hiopVectorRajaPar::min() const
{
  double* data = data_dev_;
  RAJA::ReduceMin< hiop_raja_reduce, double > minimum(std::numeric_limits<double>::max());
  RAJA::forall< hiop_raja_exec >(
    RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      minimum.min(data[i]);
    }
  );
  double ret_val = minimum.get();

#ifdef HIOP_USE_MPI
  double ret_val_g;
  int ierr=MPI_Allreduce(&ret_val, &ret_val_g, 1, MPI_DOUBLE, MPI_MIN, comm_); assert(MPI_SUCCESS==ierr);
  ret_val = ret_val_g;
#endif
  return ret_val;
}

/// Find minimum vector element for `select` pattern
double hiopVectorRajaPar::min_w_pattern(const hiopVector& select) const
{
  const hiopVectorRajaPar& sel = dynamic_cast<const hiopVectorRajaPar&>(select);
  assert(this->n_local_ == sel.n_local_);
  double* data = data_dev_;
  const double* id = sel.local_data_const();
  
  RAJA::ReduceMin< hiop_raja_reduce, double > minimum(std::numeric_limits<double>::max());
  RAJA::forall< hiop_raja_exec >(
    RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      if(id[i] == one) {
        minimum.min(data[i]);
      }
    }
  );
  double ret_val = minimum.get();

#ifdef HIOP_USE_MPI
  double ret_val_g;
  int ierr=MPI_Allreduce(&ret_val, &ret_val_g, 1, MPI_DOUBLE, MPI_MIN, comm_); assert(MPI_SUCCESS==ierr);
  ret_val = ret_val_g;
#endif
  return ret_val;
}

/// Find minimum vector element
void hiopVectorRajaPar::min( double& /* m */, int& /* index */) const
{
  assert(false && "not implemented");
}

/**
 * @brief Negate all vector elements
 * 
 * @note Consider implementing with BLAS call (<D>SCAL)
 */
void hiopVectorRajaPar::negate()
{
  double* data = data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      data[i] *= -1;
    });
}

/**
 * @brief Invert vector elements
 * 
 * @pre this[i] != 0 forall i
 * @post `this` is overwritten
 * 
 * @todo Consider having HiOp-wide `small_real` constant defined.
 */
void hiopVectorRajaPar::invert()
{
#ifdef HIOP_DEEPCHECKS
#ifndef NDEBUG
  const double small_real = 1e-35;
#endif
#endif
  double *data = data_dev_;
  RAJA::forall< hiop_raja_exec >(RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
#ifdef HIOP_DEEPCHECKS
      assert(fabs(data[i]) > small_real);
#endif
      data[i] = one/data[i];
    });
}

/**
 * @brief Sum all selected log(this[i])
 * 
 * @pre `this` and `select` have same partitioning.
 * @pre Selected elements of `this` are > 0.
 * @post `this` and `select` are not modified
 * 
 * @warning This is local method only!
 */
double hiopVectorRajaPar::logBarrier_local(const hiopVector& select) const
{
  const hiopVectorRajaPar& sel = dynamic_cast<const hiopVectorRajaPar&>(select);
  assert(this->n_local_ == sel.n_local_);

  double* data = data_dev_;
  const double* id = sel.local_data_const();
  RAJA::ReduceSum< hiop_raja_reduce, double > sum(0.0);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
		RAJA_LAMBDA(RAJA::Index_type i)
    {
#ifdef HIOP_DEEPCHECKS
      assert(id[i] == one || id[i] == zero);
#endif
      if(id[i] == one)
        sum += std::log(data[i]);
		});

  return sum.get();
}

/**
 * @brief Sum all elements
 */
double hiopVectorRajaPar::sum_local() const
{
  double* data = data_dev_;
  RAJA::ReduceSum< hiop_raja_reduce, double > sum(0.0);
  RAJA::forall< hiop_raja_exec >(
    RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      sum += data[i];
    }
  );

  return sum.get();
}

/**
 * @brief Sum all selected log(this[i])
 * 
 * @pre `this`, `xvec` and `select` have same partitioning.
 * @pre xvec[i] != 0 forall i
 * @post `xvec` and `select` are not modified
 */
void hiopVectorRajaPar::addLogBarrierGrad(
  double alpha,
  const hiopVector& xvec,
  const hiopVector& select)
{
  const hiopVectorRajaPar& x = dynamic_cast<const hiopVectorRajaPar&>(xvec);
  const hiopVectorRajaPar& sel = dynamic_cast<const hiopVectorRajaPar&>(select);  
#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == x.n_local_);
  assert(n_local_ == sel.n_local_);
#endif
  double* data = data_dev_;
  const double* xd = x.local_data_const();
  const double* id = sel.local_data_const();
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) 
    {
      if (id[i] == 1.0) 
        data[i] += alpha/xd[i];
    });
}

/**
 * @brief Linear damping term
 * 
 * @pre `this`, `ixleft` and `ixright` have same partitioning.
 * @pre `ixleft` and `ixright` elements are 0 or 1 only.
 * @post `this`, `ixleft` and `ixright` are not modified
 * 
 * @warning This is local method only!
 */
double hiopVectorRajaPar::linearDampingTerm_local(
  const hiopVector& ixleft,
  const hiopVector& ixright,
  const double& mu,
  const double& kappa_d) const
{
  const hiopVectorRajaPar& ixl = dynamic_cast<const hiopVectorRajaPar&>(ixleft);
  const hiopVectorRajaPar& ixr = dynamic_cast<const hiopVectorRajaPar&>(ixright);  
#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == ixl.n_local_);
  assert(n_local_ == ixr.n_local_);
#endif
  const double* ld = ixl.local_data_const();
  const double* rd = ixr.local_data_const();
  double* data = data_dev_;
  RAJA::ReduceSum< hiop_raja_reduce, double > sum(zero);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
		RAJA_LAMBDA(RAJA::Index_type i)
    {
      if (ld[i] == one && rd[i] == zero)
        sum += data[i];
    });
  double term = sum.get();
  term *= mu; 
  term *= kappa_d;
  return term;
}

void hiopVectorRajaPar::addLinearDampingTerm(
  const hiopVector& ixleft,
  const hiopVector& ixright,
  const double& alpha,
  const double& ct)
{

  assert((dynamic_cast<const hiopVectorRajaPar&>(ixleft)).n_local_ == n_local_);
  assert((dynamic_cast<const hiopVectorRajaPar&>(ixright)).n_local_ == n_local_);

  const double* ixl= (dynamic_cast<const hiopVectorRajaPar&>(ixleft)).local_data_const();
  const double* ixr= (dynamic_cast<const hiopVectorRajaPar&>(ixright)).local_data_const();

  double* data = data_dev_;

  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      // y := a * x + ...
      data[i] = alpha * data[i] + ct*(ixl[i]-ixr[i]);
    });

}

/**
 * @brief Check if all elements of the vector are positive
 * 
 * @post `this` is not modified
 */
int hiopVectorRajaPar::allPositive()
{
  double* data = data_dev_;
  RAJA::ReduceMin< hiop_raja_reduce, double > minimum(one);
  RAJA::forall< hiop_raja_exec >(RAJA::RangeSegment(0, n_local_),
                                 RAJA_LAMBDA(RAJA::Index_type i)
                                 {
                                   minimum.min(data[i]);
                                 });
  int allPos = minimum.get() > zero ? 1 : 0;

#ifdef HIOP_USE_MPI
  int allPosG;
  int ierr=MPI_Allreduce(&allPos, &allPosG, 1, MPI_INT, MPI_MIN, comm_); assert(MPI_SUCCESS==ierr);
  return allPosG;
#endif
  return allPos;
}

/**
 * @brief Project solution into bounds
 * 
 * @pre `this`, `xlo`, `ixl`, `xup` and `ixu` have same partitioning.
 * @pre `ixl` and `ixu` elements are 0 or 1 only.
 * @post `xlo`, `ixl`, `xup` and `ixu` are not modified
 * 
 * @warning This is local method only!
 */
bool hiopVectorRajaPar::projectIntoBounds_local(
  const hiopVector& xlo, 
  const hiopVector& ixl,
	const hiopVector& xup,
  const hiopVector& ixu,
	double kappa1,
  double kappa2)
{
  const hiopVectorRajaPar& xl = dynamic_cast<const hiopVectorRajaPar&>(xlo);
  const hiopVectorRajaPar& il = dynamic_cast<const hiopVectorRajaPar&>(ixl);
  const hiopVectorRajaPar& xu = dynamic_cast<const hiopVectorRajaPar&>(xup);
  const hiopVectorRajaPar& iu = dynamic_cast<const hiopVectorRajaPar&>(ixu);

#ifdef HIOP_DEEPCHECKS
  assert(xl.n_local_ == n_local_);
  assert(il.n_local_ == n_local_);
  assert(xu.n_local_ == n_local_);
  assert(iu.n_local_ == n_local_);
#endif

  const double* xld = xl.local_data_const();
  const double* ild = il.local_data_const();
  const double* xud = xu.local_data_const();
  const double* iud = iu.local_data_const();
  double* xd = data_dev_; 
  // Perform preliminary check to see of all upper value
  RAJA::ReduceMin< hiop_raja_reduce, double > minimum(one);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i)
      {
        minimum.min(xud[i] - xld[i]);
      });
  if (minimum.get() < zero) 
    return false;

  const double small_real = std::numeric_limits<double>::min() * 100;

  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      double aux  = zero;
      double aux2 = zero;
      if(ild[i] != zero && iud[i] != zero)
      {
        aux = kappa2*(xud[i] - xld[i]) - small_real;
        aux2 = xld[i] + fmin(kappa1 * fmax(one, fabs(xld[i])), aux);
        if(xd[i] < aux2)
        {
          xd[i] = aux2;
        }
        else
        {
          aux2 = xud[i] - fmin(kappa1 * fmax(one, fabs(xud[i])), aux);
          if(xd[i] > aux2)
          {
            xd[i] = aux2;
          }
        }
#ifdef HIOP_DEEPCHECKS
      assert(xd[i] > xld[i] && xd[i] < xud[i] && "this should not happen -> HiOp bug");
#endif
      }
      else
      {
        if(ild[i] != zero)
          xd[i] = fmax(xd[i], xld[i] + kappa1*fmax(one, fabs(xld[i])) - small_real);
        else 
          if(iud[i] != zero)
            xd[i] = fmin(xd[i], xud[i] - kappa1*fmax(one, fabs(xud[i])) - small_real);
          else { /*nothing for free vars  */ }
      }
    });
  return true;
}

/**
 * @brief max{a\in(0,1]| x+ad >=(1-tau)x}
 * 
 * @pre `this` and `dvec` have same partitioning.
 * @post `this` and `dvec` are not modified
 * 
 * @warning This is local method only!
 */
double hiopVectorRajaPar::fractionToTheBdry_local(const hiopVector& dvec, const double& tau) const
{
  const hiopVectorRajaPar& d = dynamic_cast<const hiopVectorRajaPar&>(dvec);
#ifdef HIOP_DEEPCHECKS
  assert(d.n_local_ == n_local_);
  assert(tau > 0);
  assert(tau < 1); // TODO: per documentation above it should be tau <= 1 (?).
#endif

  const double* dd = d.local_data_const();
  const double* xd = data_dev_;

  RAJA::ReduceMin< hiop_raja_reduce, double > minimum(one);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      if(dd[i] >= zero)
        return;
#ifdef HIOP_DEEPCHECKS
      assert(xd[i] > zero);
#endif
      minimum.min(-tau*xd[i]/dd[i]);
    });
  return minimum.get();
}

/**
 * @brief max{a\in(0,1]| x+ad >=(1-tau)x} with pattern select
 * 
 * @pre `this`, `select` and `dvec` have same partitioning.
 * @pre Elements of `select` are either 0 or 1.
 * @post `this`, `select` and `dvec` are not modified
 * 
 * @warning This is local method only!
 */
double hiopVectorRajaPar::fractionToTheBdry_w_pattern_local(
  const hiopVector& dvec,
  const double& tau, 
  const hiopVector& select) const
{
  const hiopVectorRajaPar& d = dynamic_cast<const hiopVectorRajaPar&>(dvec);
  const hiopVectorRajaPar& s = dynamic_cast<const hiopVectorRajaPar&>(select);

#ifdef HIOP_DEEPCHECKS
  assert(d.n_local_ == n_local_);
  assert(s.n_local_ == n_local_);
  assert(tau>0);
  assert(tau<1);
#endif
  const double* dd = d.local_data_const();
  const double* xd = data_dev_;
  const double* id = s.local_data_const();

  RAJA::ReduceMin< hiop_raja_reduce, double > aux(one);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      assert(id[i] == one || id[i] == zero);
      if(dd[i] < 0 && id[i] == one)
      {
#ifdef HIOP_DEEPCHECKS
        assert(xd[i] > 0);
#endif
        aux.min(-tau*xd[i]/dd[i]);
      }
    });
  return aux.get();
}

/**
 * @brief Set elements of `this` to zero based on `select`.
 * 
 * @pre `this` and `select` have same partitioning.
 * @pre Elements of `select` are either 0 or 1.
 * @post `select` is not modified
 */
void hiopVectorRajaPar::selectPattern(const hiopVector& select)
{
  const hiopVectorRajaPar& s = dynamic_cast<const hiopVectorRajaPar&>(select);
#ifdef HIOP_DEEPCHECKS
  assert(s.n_local_==n_local_);
#endif

  double* data = data_dev_;
  double* sd = s.data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) {
      if(sd[i] == zero)
        data[i] = zero;
    });
}

/**
 * @brief Checks if `this` matches nonzero pattern of `select`.
 * 
 * @pre `this` and `select` have same partitioning.
 * @pre Elements of `select` are either 0 or 1.
 * @post `select` is not modified
 */
bool hiopVectorRajaPar::matchesPattern(const hiopVector& pattern)
{  
  const hiopVectorRajaPar& p = dynamic_cast<const hiopVectorRajaPar&>(pattern);

#ifdef HIOP_DEEPCHECKS
  assert(p.n_local_==n_local_);
#endif

  double* data = data_dev_;
  double* pd = p.data_dev_;
  RAJA::ReduceSum<hiop_raja_reduce, int> sum(0);
  RAJA::forall<hiop_raja_exec>( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      sum += (data[i] != 0.0 && pd[i] == 0.0);
    });
  int mismatch = sum.get();

#ifdef HIOP_USE_MPI
  int mismatch_glob = mismatch;
  int ierr = MPI_Allreduce(&mismatch, &mismatch_glob, 1, MPI_INT, MPI_SUM, comm_);
  assert(MPI_SUCCESS==ierr);
  return (mismatch_glob == 0);
#endif
  return (mismatch == 0);
}

/**
 * @brief Checks if selected elements of `this` are positive.
 * 
 * @pre `this` and `select` have same partitioning.
 * @pre Elements of `select` are either 0 or 1.
 * @post `select` is not modified
 */
int hiopVectorRajaPar::allPositive_w_patternSelect(const hiopVector& wvec)
{
  const hiopVectorRajaPar& w = dynamic_cast<const hiopVectorRajaPar&>(wvec);

#ifdef HIOP_DEEPCHECKS
  assert(w.n_local_ == n_local_);
#endif 

  const double* wd = w.local_data_const();
  const double* data = data_dev_;
  RAJA::ReduceSum< hiop_raja_reduce, int > sum(0);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) 
    {
      if(wd[i] != zero && data[i] <= zero)
        sum += 1;
    });
  int allPos = (sum.get() == 0);
  
#ifdef HIOP_USE_MPI
  int allPosG;
  int ierr = MPI_Allreduce(&allPos, &allPosG, 1, MPI_INT, MPI_MIN, comm_);
  assert(MPI_SUCCESS==ierr);
  return allPosG;
#endif  
  return allPos;
}

/**
 * @brief Adjusts duals.
 * 
 * @pre `this`, `xvec` and `ixvec` have same partitioning.
 * @pre Elements of `ixvec` are either 0 or 1.
 * @post `xvec` and `ixvec` are not modified
 * 
 * @note Implementation probably inefficient.
 */
void hiopVectorRajaPar::adjustDuals_plh(
  const hiopVector& xvec, 
  const hiopVector& ixvec,
  const double& mu,
  const double& kappa)
{
  const hiopVectorRajaPar& x  = dynamic_cast<const hiopVectorRajaPar&>(xvec) ;
  const hiopVectorRajaPar& ix = dynamic_cast<const hiopVectorRajaPar&>(ixvec);
#ifdef HIOP_DEEPCHECKS
  assert(x.n_local_==n_local_);
  assert(ix.n_local_==n_local_);
#endif
  const double* xd =  x.local_data_const();
  const double* id = ix.local_data_const();
  double* z = data_dev_; //the dual

  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      double a,b;
      // preemptive loop to reduce number of iterations?
      if(id[i] == 1.) {
          // precompute a and b in another loop?
          a = mu/xd[i];
          b = a/kappa;
          a = a*kappa;
          // Necessary conditionals
          if(z[i]<b) 
            z[i]=b;
          else //z[i]>=b
            if(a<=b) 
              z[i]=b;
            else //a>b
              if(a<z[i])
                z[i]=a;
          // - - - - 
          //else a>=z[i] then *z=*z (z[i] does not need adjustment)
      }
    });
}

/**
 * @brief Returns true if any element of `this` is NaN.
 * 
 * @post `this` is not modified
 * 
 * @warning This is local method only!
 */
bool hiopVectorRajaPar::isnan_local() const
{
  double* data = data_dev_;
  RAJA::ReduceSum< hiop_raja_reduce, int > any(0);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      if(std::isnan(data[i]))
        any += 1;
    });
  return any.get();
}

/**
 * @brief Returns true if any element of `this` is Inf.
 * 
 * @post `this` is not modified
 * 
 * @warning This is local method only!
 */
bool hiopVectorRajaPar::isinf_local() const
{
  double* data = data_dev_;
  RAJA::ReduceSum< hiop_raja_reduce, int > any(0);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      if(std::isinf(data[i]))
        any += 1;
    });
  return any.get();
}

/**
 * @brief Returns true if all elements of `this` are finite.
 * 
 * @post `this` is not modified
 * 
 * @warning This is local method only!
 */
bool hiopVectorRajaPar::isfinite_local() const
{
  double* data = data_dev_;
  RAJA::ReduceMin< hiop_raja_reduce, int > smallest(1);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      if(!std::isfinite(data[i]))
        smallest.min(0);
    });
  return smallest.get();
}

/**
 * @brief Prints vector data to a file in Matlab format.
 * 
 * @pre Vector data was moved from the memory space to the host mirror.
 */
void hiopVectorRajaPar::print(FILE* file, const char* msg/*=NULL*/, int max_elems/*=-1*/, int rank/*=-1*/) const
{
  int myrank=0, numranks=1; 
#ifdef HIOP_USE_MPI
  if(rank >= 0) {
    int err = MPI_Comm_rank(comm_, &myrank); assert(err==MPI_SUCCESS);
    err = MPI_Comm_size(comm_, &numranks); assert(err==MPI_SUCCESS);
  }
#endif
  if(myrank == rank || rank == -1)
  {
    if(max_elems>n_local_)
      max_elems=n_local_;

    if(NULL==msg)
    {
      std::stringstream ss;
      ss << "vector of size " << n_ << ", printing " << max_elems << " elems ";
      if(numranks>1) {
        ss << "(on rank=" << myrank << ")";
      }
      else {
        ss << "(serial)";
      }
      ss << "\n";
      fprintf(file, "%s", ss.str().c_str());
    }
    else
    {
      fprintf(file, "%s ", msg);
    }    
    fprintf(file, "=[");
    max_elems = max_elems >= 0 ? max_elems : n_local_;
    for(int it=0; it<max_elems; it++)
      fprintf(file, "%22.16e ; ", data_host_[it]);
    fprintf(file, "];\n");
  }
}

void hiopVectorRajaPar::copyToDev()
{
  if(data_dev_ == data_host_)
    return;
  auto& resmgr = umpire::ResourceManager::getInstance();
  resmgr.copy(data_dev_, data_host_);
}

void hiopVectorRajaPar::copyFromDev()
{
  if(data_dev_ == data_host_)
    return;
  auto& resmgr = umpire::ResourceManager::getInstance();
  resmgr.copy(data_host_, data_dev_);
}

void hiopVectorRajaPar::copyToDev() const
{
  if(data_dev_ == data_host_)
    return;
  auto& resmgr = umpire::ResourceManager::getInstance();
  double* data_dev = const_cast<double*>(data_dev_);
  resmgr.copy(data_dev, data_host_);
}

void hiopVectorRajaPar::copyFromDev() const
{
  if(data_dev_ == data_host_)
    return;
  auto& resmgr = umpire::ResourceManager::getInstance();
  double* data_host = const_cast<double*>(data_host_);
  resmgr.copy(data_host, data_dev_);
}

size_type hiopVectorRajaPar::numOfElemsLessThan(const double &val) const
{  
  double* data = data_dev_;
  RAJA::ReduceSum<hiop_raja_reduce, size_type> sum(0);
  RAJA::forall<hiop_raja_exec>( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      sum += (data[i]<val);
    });

  size_type nrm = sum.get();

#ifdef HIOP_USE_MPI
  size_type nrm_global;
  int ierr = MPI_Allreduce(&nrm, &nrm_global, 1, MPI_HIOP_SIZE_TYPE, MPI_SUM, comm_);
  assert(MPI_SUCCESS == ierr);
  nrm = nrm_global;
#endif

  return nrm;
}

size_type hiopVectorRajaPar::numOfElemsAbsLessThan(const double &val) const
{  
  double* data = data_dev_;
  RAJA::ReduceSum<hiop_raja_reduce, size_type> sum(0);
  RAJA::forall<hiop_raja_exec>( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      sum += static_cast<size_type>(fabs(data[i]) < val);
    });

  size_type nrm = sum.get();

#ifdef HIOP_USE_MPI
  size_type nrm_global;
  int ierr = MPI_Allreduce(&nrm, &nrm_global, 1, MPI_HIOP_SIZE_TYPE, MPI_SUM, comm_);
  assert(MPI_SUCCESS == ierr);
  nrm = nrm_global;
#endif

  return nrm;
}

void hiopVectorRajaPar::set_array_from_to(hiopInterfaceBase::NonlinearityType* arr, 
                                          const int start, 
                                          const int end, 
                                          const hiopInterfaceBase::NonlinearityType* arr_src,
                                          const int start_src) const
{
  assert(end <= n_local_ && start <= end && start >= 0 && start_src >= 0);

  // If there is nothing to copy, return.
  if(end - start == 0)
    return;
  
  RAJA::forall< hiop_raja_exec >(
    RAJA::RangeSegment(0, end-start),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      arr[start+i] = arr_src[start_src+i];
    }
  );

}

void hiopVectorRajaPar::set_array_from_to(hiopInterfaceBase::NonlinearityType* arr, 
                                          const int start, 
                                          const int end, 
                                          const hiopInterfaceBase::NonlinearityType arr_src) const
{
  assert(end <= n_local_ && start <= end && start >= 0);

  // If there is nothing to copy, return.
  if(end - start == 0)
    return;

  auto& rm = umpire::ResourceManager::getInstance();
  RAJA::forall< hiop_raja_exec >(
    RAJA::RangeSegment(start, end),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      arr[i] = arr_src;
    }
  );      
}

bool hiopVectorRajaPar::is_equal(const hiopVector& vec) const
{
#ifdef HIOP_DEEPCHECKS
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(vec);
  assert(v.n_local_ == n_local_);
#endif 

  const double* data_v = vec.local_data_const();
  const double* data = data_dev_;
  RAJA::ReduceSum< hiop_raja_reduce, int > sum(0);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) 
    {
      if(data[i]!=data_v[i]) {
        sum += 1;        
      }
    });
  int all_equal = (sum.get() == 0);
  
#ifdef HIOP_USE_MPI
  int all_equalG;
  int ierr = MPI_Allreduce(&all_equal, &all_equalG, 1, MPI_INT, MPI_MIN, comm_);
  assert(MPI_SUCCESS==ierr);
  return all_equalG;
#endif  
  return all_equal;
}

} // namespace hiop
